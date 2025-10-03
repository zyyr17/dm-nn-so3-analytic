import apsw #way faster than sqlite3
import math
import numpy as np
import io
import os
import multiprocessing
STORAGE_DTYPE = np.float64
STORAGE_BYTES = np.dtype(STORAGE_DTYPE).itemsize  # 8
dtype = np.float64
class HamiltonianDatabase:
    def __init__(self, filename, flags=apsw.SQLITE_OPEN_READONLY):
        self.db = filename
        self.connections = {} #allow multiple connections (needed for multi-threading)
        self._open(flags=flags) #creates the database if it doesn't exist yet

    def __len__(self):
        cursor = self._get_connection(flags=apsw.SQLITE_OPEN_READONLY).cursor()
        row = cursor.execute('SELECT * FROM metadata WHERE id=0').fetchone()
        print(f"[✓] __len__: metadata row = {row}")
        if row is None:
            return 0
        return row[1]  # This is the correct N value
    
    def __getitem__(self, idx):
        cursor = self._get_connection(flags=apsw.SQLITE_OPEN_READONLY).cursor()
        if type(idx) == list: #for batched data retrieval
            data = cursor.execute('''SELECT * FROM data WHERE id IN ('''+str(idx)[1:-1]+')').fetchall()
            return [self._unpack_data_tuple(i) for i in data]
        else:
            data = cursor.execute('''SELECT * FROM data WHERE id='''+str(idx)).fetchone()
            return self._unpack_data_tuple(data)
        
    def _unpack_data_tuple(self, data):
        # positions R: shape (N,3)
        N = len(data[1]) // (STORAGE_BYTES * 3)
        R = self._deblob(data[1], dtype=STORAGE_DTYPE, shape=(N,3))
    
        # energy (SQLite REAL is fine)
        e_raw = data[2]
        if e_raw is None:
            E = np.array([0.0], dtype=STORAGE_DTYPE)
        elif isinstance(e_raw, (bytes, bytearray, memoryview)):
            E = np.frombuffer(e_raw, STORAGE_DTYPE, count=1)  # decode float64 from blob
        else:
            E = np.array([float(e_raw)], dtype=STORAGE_DTYPE)

    
        # infer Norb from P
        norb_bytes = len(data[4]) // STORAGE_BYTES
        Norb = int(math.isqrt(norb_bytes))  # safer than sqrt on floats
    
        # H represents the Fock matrix in DFT; the matrix G in HF
        
        H = self._deblob(data[3], dtype=STORAGE_DTYPE, shape=(Norb, Norb))
    
        P = self._deblob(data[4], dtype=STORAGE_DTYPE, shape=(Norb, Norb))
        S = self._deblob(data[5], dtype=STORAGE_DTYPE, shape=(Norb, Norb))
        C = self._deblob(data[6], dtype=STORAGE_DTYPE, shape=(Norb, Norb))
        return R, E, H, P, S, C
    def add_data(self, R, E, H, P, S, C, flags=apsw.SQLITE_OPEN_READWRITE, transaction=True):
        #check that no NaN values are added
        if self._any_is_nan(R, E, H, P, S, C):
            print("encountered NaN, data is not added")
            return
        cursor = self._get_connection(flags=flags).cursor()
        #update data
        if transaction:
            #begin exclusive transaction (locks db) which is necessary
            #if database is accessed from multiple programs at once (default for safety)
            cursor.execute('''BEGIN EXCLUSIVE''') 
        try:
            length = cursor.execute('''SELECT * FROM metadata WHERE id=0''').fetchone()[-1]
            cursor.execute('''INSERT INTO data (id, R, E, H, P, S, C) VALUES (?,?,?,?,?,?,?)''', (None if length > 0 else 0, self._blob(R), None if E is None else float(E),  self._blob(H), self._blob(P), self._blob(S), self._blob(C)))

            cursor.execute('''INSERT OR REPLACE INTO metadata VALUES (?,?)''', (0, length+1))
            if transaction:
                cursor.execute('''COMMIT''') #end transaction
        except Exception as exc:
            if transaction:
                cursor.execute('''ROLLBACK''')
            raise exc
    def add_orbitals(self, Z, orbitals, flags=apsw.SQLITE_OPEN_READWRITE):
        cursor = self._get_connection(flags=flags).cursor()
        cursor.execute('''INSERT OR REPLACE INTO basisset (Z, orbitals) VALUES (?,?)''', 
                (int(Z), self._blob(orbitals)))

    def get_orbitals(self, Z):
        # still int32 in DB → derive bytes from dtype
        itembytes = np.dtype(np.int32).itemsize
        cursor = self._get_connection().cursor()
        data = cursor.execute('SELECT * FROM basisset WHERE Z=?', (int(Z),)).fetchone()
        Norb = len(data[1]) // itembytes
        return self._deblob(data[1], dtype=np.int32, shape=(Norb,))

    def _any_is_nan(self, *vals):
        nan = False
        for val in vals:
            if val is None:
                continue
            nan = nan or np.any(np.isnan(val))
        return nan

    def _blob(self, array):
        if array is None: return None
        # write as float64
        if array.dtype.kind == 'f' and array.dtype != np.float64:
            array = array.astype(np.float64)
        if array.dtype == np.int64:
            array = array.astype(np.int32)
        if not np.little_endian:
            array = array.byteswap()
        return memoryview(np.ascontiguousarray(array))

    def _deblob(self, buf, dtype, shape=None):
        if buf is None:
            return np.zeros(shape, dtype=dtype)
        arr = np.frombuffer(buf, dtype)
        if not np.little_endian:
            arr = arr.byteswap()
        if shape is not None:
            arr = arr.reshape(shape)
        return arr
    def _open(self, flags=apsw.SQLITE_OPEN_READONLY):
        newdb = not os.path.isfile(self.db)
        cursor = self._get_connection(flags=flags).cursor()
        if newdb:
            #create table to store data
            cursor.execute('''CREATE TABLE IF NOT EXISTS data
                (id INTEGER NOT NULL PRIMARY KEY,
                 R BLOB,
                 E BLOB,
                 H BLOB,
                 P BLOB,
                 S BLOB,
                 C BLOB
                )''')

            #create table to store things that are constant for the whole dataset
            cursor.execute('''CREATE TABLE IF NOT EXISTS nuclear_charges
                (id INTEGER NOT NULL PRIMARY KEY, N INTEGER, Z BLOB)''')
            cursor.execute('''INSERT OR IGNORE INTO nuclear_charges (id, N, Z) VALUES (?,?,?)''', 
                (0, 1, self._blob(np.array([0]))))
            self.N = len(self.Z)

            #create table to store the basis set convention
            cursor.execute('''CREATE TABLE IF NOT EXISTS basisset
                (Z INTEGER NOT NULL PRIMARY KEY, orbitals BLOB)''')
      
            #create table to store metadata (information about the number of entries)
            cursor.execute('''CREATE TABLE IF NOT EXISTS metadata
                (id INTEGER PRIMARY KEY, N INTEGER)''')
            cursor.execute('''INSERT OR IGNORE INTO metadata (id, N) VALUES (?,?)''', (0, 0)) #num_data

    def _get_connection(self, flags=apsw.SQLITE_OPEN_READONLY):
        '''
        This allows multiple processes to access the database at once,
        every process must have its own connection
        '''
        key = multiprocessing.current_process().name
        if key not in self.connections.keys():
            self.connections[key] = apsw.Connection(self.db, flags=flags)
            self.connections[key].setbusytimeout(300000) #5 minute timeout
        return self.connections[key]

    def add_Z(self, Z, flags=apsw.SQLITE_OPEN_READWRITE):
        cursor = self._get_connection(flags=flags).cursor()
        self.N = len(Z)
        cursor.execute('''INSERT OR REPLACE INTO nuclear_charges (id, N, Z) VALUES (?,?,?)''', 
                (0, self.N, self._blob(Z)))

    @property
    def Z(self):
        cursor = self._get_connection(flags=apsw.SQLITE_OPEN_READONLY).cursor()
        data = cursor.execute('''SELECT * FROM nuclear_charges WHERE id=0''').fetchone()
        N = data[1]
        return self._deblob(data[2], dtype=np.int32, shape=(N,))

