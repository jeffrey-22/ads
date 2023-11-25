from .config import *

import yaml, os, wget, requests, pymysql, zipfile
import dask.dataframe as dd

# This file accesses the data

"""Place commands in this file to access the data electronically. Don't remove any missing values, or deal with outliers. Make sure you have legalities correct, both intellectual property and personal data privacy rights. Beyond the legal side also think about the ethical issues around this data. """

def retreive_database_details(database_name = "property_prices"):
    general_database_details = {"database_url": config["database_url"],
                                "database_port": config["database_port"],
                                "database_username": config["database_username"],
                                "database_password": config["database_password"],
                                }
    pp_database_details = general_database_details.copy()
    pp_database_details["database_name"] = database_name
    return pp_database_details

def create_database(database_details):
    connection = pymysql.connect(
        host=database_details["database_url"],
        port=database_details["database_port"],
        user=database_details["database_username"],
        password=database_details["database_password"],
    )
    try:
        with connection.cursor() as cursor:
            database_name = database_details["database_name"]
            create_database_query = f"CREATE DATABASE IF NOT EXISTS `{database_name}` DEFAULT CHARACTER SET utf8 COLLATE utf8_bin"
            cursor.execute(create_database_query)
        connection.commit()
    except pymysql.Error as e:
        print(f"Error creating database: {e}")
    finally:
        connection.close()

def download_file(url, downloaded_pathnames = set()):
    def name_file(url):
        return str(url).split('/')[-1]
    def bar_progress(current, total, width):
        percent = int(100 * current / total)
        print(f'Current file {percent}% complete\r', end='')
    pathname = os.path.join("tmp_data", name_file(url))
    downloaded_pathnames.add(pathname)
    if (not os.path.isfile(pathname)):
        wget.download(url, pathname, bar=bar_progress)
    return os.path.isfile(pathname)

def download_file_requests(url, output_filename):
    # Slower than wget but at acceptable rate
    try:
        response = requests.get(url)
        response.raise_for_status()
        with open(output_filename, 'wb') as file:
            file.write(response.content)
    except requests.HTTPError as http_err:
        print(f"HTTP Error: {http_err}")
    except Exception as err:
        print(f"Error during download: {err}")

def download_price_data():
    downloaded_pathnames = set()
    url = 'http://prod.publicdata.landregistry.gov.uk.s3-website-eu-west-1.amazonaws.com/pp-2022.csv'
    assert(download_file(url, downloaded_pathnames))
    for year in range(1995, 2021 + 1):
        for part in range(1, 2 + 1):
            url = f'http://prod.publicdata.landregistry.gov.uk.s3-website-eu-west-1.amazonaws.com/pp-{year}-part{part}.csv'
            assert(download_file(url, downloaded_pathnames))
    return downloaded_pathnames

def create_connection(database_details = retreive_database_details()):
    try:
        connection = pymysql.connect(
            host=database_details["database_url"],
            port=database_details["database_port"],
            user=database_details["database_username"],
            password=database_details["database_password"],
            db=database_details["database_name"],
            local_infile=1,
        )
    except Exception as e:
        print(f"Error connecting to the MariaDB Server: {e}")
    return connection

class DatabaseConnection:
    _connection = None

    def __init__(self):
        if DatabaseConnection._connection is None:
            DatabaseConnection._connection = create_connection()

    @staticmethod
    def get_connection():
        if DatabaseConnection._connection is None:
            DatabaseConnection._connection = create_connection()
        return DatabaseConnection._connection

    @staticmethod
    def close_connection():
        if DatabaseConnection._connection is not None:
            DatabaseConnection._connection.close()
            DatabaseConnection._connection = None

    def __del__(self):
        self.close_connection()

def create_pp_table(conn):
    with conn.cursor() as cursor:
        # source of creation: https://github.com/dalepotter/uk_property_price_data/blob/master/create_db.sql
        query = "DROP TABLE IF EXISTS `pp_data`;"
        cursor.execute(query)
        query = '''
        CREATE TABLE IF NOT EXISTS `pp_data` (
        `transaction_unique_identifier` tinytext COLLATE utf8_bin NOT NULL,
        `price` int(10) unsigned NOT NULL,
        `date_of_transfer` date NOT NULL,
        `postcode` varchar(8) COLLATE utf8_bin NOT NULL,
        `property_type` varchar(1) COLLATE utf8_bin NOT NULL,
        `new_build_flag` varchar(1) COLLATE utf8_bin NOT NULL,
        `tenure_type` varchar(1) COLLATE utf8_bin NOT NULL,
        `primary_addressable_object_name` tinytext COLLATE utf8_bin NOT NULL,
        `secondary_addressable_object_name` tinytext COLLATE utf8_bin NOT NULL,
        `street` tinytext COLLATE utf8_bin NOT NULL,
        `locality` tinytext COLLATE utf8_bin NOT NULL,
        `town_city` tinytext COLLATE utf8_bin NOT NULL,
        `district` tinytext COLLATE utf8_bin NOT NULL,
        `county` tinytext COLLATE utf8_bin NOT NULL,
        `ppd_category_type` varchar(2) COLLATE utf8_bin NOT NULL,
        `record_status` varchar(2) COLLATE utf8_bin NOT NULL,
        `db_id` bigint(20) unsigned NOT NULL
        ) DEFAULT CHARSET=utf8 COLLATE=utf8_bin AUTO_INCREMENT=1 ;
        '''
        cursor.execute(query)
    conn.commit()

def upload_files_to_table(conn, pathnames, tablename, remove_file_as_uploading = False, ignore_first_row = False):
    files_uploaded = 0
    for filename in pathnames:
        upload_filename = filename.replace('\\', '\\\\')
        with conn.cursor() as cursor:
            query = f"                                                         \
            LOAD DATA LOCAL INFILE '{upload_filename}' INTO TABLE `{tablename}`\
            FIELDS TERMINATED BY ',' OPTIONALLY ENCLOSED by '\"'               \
            LINES STARTING BY '' TERMINATED BY '\\n'                           \
            "
            if ignore_first_row:
                query += "IGNORE 1 LINES;"
            else:
                query += ";"
            cursor.execute(query)
            if remove_file_as_uploading:
                os.remove(upload_filename)
            files_uploaded += 1
            print(f"{100.0 * files_uploaded / len(pathnames)}% of files uploaded\r", end="")
    conn.commit()

def setup_pp_table(conn):
    with conn.cursor() as cursor:
        query = '''
        ALTER TABLE `pp_data`
        MODIFY db_id bigint(20) unsigned NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=1,
        ADD PRIMARY KEY (`db_id`);
        '''
        cursor.execute(query)
    conn.commit()

def select_count(conn, tablename):
    cur = conn.cursor()
    cur.execute(f'SELECT COUNT(*) FROM {tablename}')
    rows = cur.fetchall()
    return rows

def create_postcode_table(conn):
    with conn.cursor() as cursor:
        query = "DROP TABLE IF EXISTS `postcode_data`;"
        cursor.execute(query)
        query = '''
        CREATE TABLE IF NOT EXISTS `postcode_data` (
        `postcode` varchar(8) COLLATE utf8_bin NOT NULL,
        `status` enum('live','terminated') NOT NULL,
        `usertype` enum('small', 'large') NOT NULL,
        `easting` int unsigned,
        `northing` int unsigned,
        `positional_quality_indicator` int NOT NULL,
        `country` enum('England', 'Wales', 'Scotland', 'Northern Ireland', 'Channel Islands', 'Isle of Man') NOT NULL,
        `latitude` decimal(11,8) NOT NULL,
        `longitude` decimal(10,8) NOT NULL,
        `postcode_no_space` tinytext COLLATE utf8_bin NOT NULL,
        `postcode_fixed_width_seven` varchar(7) COLLATE utf8_bin NOT NULL,
        `postcode_fixed_width_eight` varchar(8) COLLATE utf8_bin NOT NULL,
        `postcode_area` varchar(2) COLLATE utf8_bin NOT NULL,
        `postcode_district` varchar(4) COLLATE utf8_bin NOT NULL,
        `postcode_sector` varchar(6) COLLATE utf8_bin NOT NULL,
        `outcode` varchar(4) COLLATE utf8_bin NOT NULL,
        `incode` varchar(3)  COLLATE utf8_bin NOT NULL,
        `db_id` bigint(20) unsigned NOT NULL
        ) DEFAULT CHARSET=utf8 COLLATE=utf8_bin;
        '''
        cursor.execute(query)
    conn.commit()

def download_postcode_data(url = 'https://www.getthedata.com/downloads/open_postcode_geo.csv.zip'):
    postcode_zipname = os.path.join("tmp_data", "open_postcode_geo.csv.zip")
    postcode_filename = os.path.join("tmp_data", "open_postcode_geo.csv")
    extract_path = os.path.dirname(postcode_filename)
    download_file_requests(url, postcode_zipname)

    def unzip_file(zip_path):
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)

    unzip_file(postcode_zipname)
    return postcode_filename

def setup_postcode_table(conn):
    with conn.cursor() as cursor:
        query = '''
        ALTER TABLE `postcode_data`
        MODIFY `db_id` bigint(20) unsigned NOT NULL AUTO_INCREMENT,AUTO_INCREMENT=1,
        ADD PRIMARY KEY (`db_id`);
        '''
        cursor.execute(query)
    conn.commit()

def create_prices_coordinates_table(conn):
    with conn.cursor() as cursor:
        query = "DROP TABLE IF EXISTS `prices_coordinates_data`;"
        cursor.execute(query)
        query = '''
        CREATE TABLE IF NOT EXISTS `prices_coordinates_data` (
            `price` int(10) unsigned NOT NULL,
            `date_of_transfer` date NOT NULL,
            `postcode` varchar(8) COLLATE utf8_bin NOT NULL,
            `property_type` varchar(1) COLLATE utf8_bin NOT NULL,
            `new_build_flag` varchar(1) COLLATE utf8_bin NOT NULL,
            `tenure_type` varchar(1) COLLATE utf8_bin NOT NULL,
            `locality` tinytext COLLATE utf8_bin NOT NULL,
            `town_city` tinytext COLLATE utf8_bin NOT NULL,
            `district` tinytext COLLATE utf8_bin NOT NULL,
            `county` tinytext COLLATE utf8_bin NOT NULL,
            `country` enum('England', 'Wales', 'Scotland', 'Northern Ireland', 'Channel Islands', 'Isle of Man') NOT NULL,
            `latitude` decimal(11,8) NOT NULL,
            `longitude` decimal(10,8) NOT NULL,
            `db_id` bigint(20) unsigned NOT NULL
        ) DEFAULT CHARSET=utf8 COLLATE=utf8_bin AUTO_INCREMENT=1 ;
        '''
        cursor.execute(query)
    conn.commit()

def join_two_tables(price_table_pathname, postcode_table_pathname, joined_table_pathnames = set(), overwrite = True):
    output_pathname = price_table_pathname.split('.')[0] + "-joined.csv"
    if (overwrite or not (os.path.exists(output_pathname))):
        column_names = [
            'transaction_unique_identifier',
            'price',
            'date_of_transfer',
            'postcode',
            'property_type',
            'new_build_flag',
            'tenure_type',
            'primary_addressable_object_name',
            'secondary_addressable_object_name',
            'street',
            'locality',
            'town_city',
            'district',
            'county',
            'ppd_category_type',
            'record_status'
        ]
        dtype_dict = {col: str for col in column_names}
        table1 = dd.read_csv(price_table_pathname, header=None, names=column_names, dtype=dtype_dict)
        table1 = table1[[
            'postcode',
            'price',
            'date_of_transfer',
            'property_type',
            'new_build_flag',
            'tenure_type',
            'locality',
            'town_city',
            'district',
            'county'
        ]]
        column_names = [
            'postcode',
            'status',
            'usertype',
            'easting',
            'northing',
            'positional_quality_indicator',
            'country',
            'latitude',
            'longitude',
            'postcode_no_space',
            'postcode_fixed_width_seven',
            'postcode_fixed_width_eight',
            'postcode_area',
            'postcode_district',
            'postcode_sector',
            'outcode',
            'incode'
        ]
        dtype_dict = {col: str for col in column_names}
        table2 = dd.read_csv(postcode_table_pathname, header=None, names=column_names, dtype=dtype_dict)
        table2 = table2[[
            'postcode',
            'country',
            'latitude',
            'longitude'
        ]]
        result = dd.merge(table1, table2, on='postcode', how='inner')
        result = result[[
            'price',
            'date_of_transfer',
            'postcode',
            'property_type',
            'new_build_flag',
            'tenure_type',
            'locality',
            'town_city',
            'district',
            'county',
            'country',
            'latitude',
            'longitude'
        ]]
        result.to_csv(output_pathname, index=False, single_file=True)
    joined_table_pathnames.add(output_pathname)
    return joined_table_pathnames

def join_all_tables(price_table_pathnames, postcode_table_pathname, joined_table_pathnames = set(), overwrite = True):
    step = 0
    for price_table_pathname in price_table_pathnames:
        joined_table_pathnames = join_two_tables(price_table_pathname, postcode_table_pathname, joined_table_pathnames, overwrite)
        step += 1
    return joined_table_pathnames

def setup_prices_coordinates_table(conn):
    with conn.cursor() as cursor:
        query = '''
        ALTER TABLE `prices_coordinates_data`
        MODIFY `db_id` bigint(20) unsigned NOT NULL AUTO_INCREMENT,AUTO_INCREMENT=1,
        ADD PRIMARY KEY (`db_id`);
        '''
        cursor.execute(query)
    conn.commit()