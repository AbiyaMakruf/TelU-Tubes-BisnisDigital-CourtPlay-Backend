import os
import logging
import uuid
from supabase import create_client, Client
from dotenv import load_dotenv

# URL dan KEY Supabase
load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# Inisialisasi Supabase Client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Konfigurasi Log
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get(table_name, columns = "*", filters = None, is_single = None):
    """
    Fungsi GET yang umum untuk Supabase.

    :param table_name: Nama tabel (contoh: 'projects', 'users').
    :param columns: Kolom yang akan diambil (contoh: 'id, name' atau '*').
    :param filters: Kondisi filter dalam bentuk dictionary (contoh: {'id': user_id, 'is_active': True}).
                    Filter saat ini hanya mendukung `eq`.
    :param is_single: Jika True, tambahkan `.single()` untuk mengambil satu baris saja.
    :returns: Data hasil query atau None jika terjadi error.
    """
    try:
        query = supabase.table(table_name).select(columns)

        if filters:
            for col, val in filters.items():
                query = query.eq(col, val)
        
        if is_single:
            query = query.single()

        response = query.execute()

        if response.data:
            logger.info("Succes get data")
        else:
            logger.warning("Get data from supabase empty.")

        return response.data
    except Exception as e:
        logger.error(f"Get data from supabase. {e}")
        return None

def update(table_name, data, filters):
    """
    Fungsi UPDATE yang umum untuk Supabase.

    :param table_name: Nama tabel.
    :param data: Dictionary data yang akan diperbarui (kolom: nilai).
    :param filters: Kondisi filter dalam bentuk dictionary (hanya mendukung `eq` saat ini).
    :returns: Data hasil update atau None jika error.
    """
    try:
        query = supabase.table(table_name).update(data)

        for col, val in filters.items():
            query = query.eq(col, val)

        response = query.execute()

        if response:
            logger.info("Success update data")

    except Exception as e:
        logger.error(f"Update data to supabase. {e}")

def delete(table_name, filters):
    """
    Fungsi DELETE yang umum untuk Supabase.
    """
    try:
        query = supabase.table(table_name).delete()

        for col, val in filters.items():
            query = query.eq(col, val)

        response = query.execute()

        if response:
            logger.info("Success delete data")
    except Exception as e:
        logger.error(f"Delete data from supabase. {e}")

def insert(table_name, data):
    """
    Fungsi INSERT yang umum untuk Supabase.
    """
    try:
        if 'id' not in data:
            data['id'] = str(uuid.uuid4())

        response = supabase.table(table_name).insert(data).execute()

        if response:
            logger.info("Success insert data")
        return response.data
    except Exception as e:
        logger.error(f"Insert data to supabase. {e}")

def test_supabase():
    """
    Test semua fungsi CRUD di Supabase.
    """
    # Test fungsi insert
    response = insert("projects", {"user_id": "a01ca3a9-345c-45ad-84cd-366f36af0c96", "project_details_id": "a01cd78d-bcb7-474e-80cd-e36c89dd2add", "project_name": "TEST FROM PYTHON", "upload_date":"now()", "created_at":"now()","updated_at":"now()"})
    PK = response[0]["id"]

    # Test fungsi get
    print(get("projects", "id, user_id, project_details_id, is_mailed", {"id": f"{PK}"}, False))

    # Test fungsi update
    update("projects", {"is_mailed": True}, {"id": f"{PK}"})

    # Test fungsi get setelah update
    print(get("projects", "is_mailed", {"id": f"{PK}"}, False))

    # Test fungsi delete
    delete("projects", {"id": f"{PK}"})

# test_supabase()