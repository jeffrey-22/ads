import os, pytest, csv, pandas as pd
from pathlib import Path

def test_wget_small_download(access_module):
    os.makedirs("tmp_data", exist_ok=True)
    url = 'https://file-examples.com/storage/fe19e15eac6560f8c936c41/2017/10/file_example_JPG_100kB.jpg'
    downloaded_pathnames = set()
    assert access_module.download_file(url, downloaded_pathnames)
    item = downloaded_pathnames.pop()
    assert len(downloaded_pathnames) == 0
    assert item == os.path.join('tmp_data', 'file_example_JPG_100kB.jpg')
    assert os.path.isfile(item)
    os.remove(item)
    os.rmdir("tmp_data")

def test_requests_small_download(access_module):
    os.makedirs("tmp_data", exist_ok=True)
    url = 'https://file-examples.com/storage/fe19e15eac6560f8c936c41/2017/10/file_example_JPG_100kB.jpg'
    path = os.path.join('tmp_data', 'file_example_JPG_100kB.jpg')
    access_module.download_file_requests(url, path)
    assert os.path.isfile(path)
    os.remove(path)
    os.rmdir("tmp_data")

@pytest.mark.slow_locally
def test_download_price_data(access_module):
    pass
    os.makedirs("tmp_data", exist_ok=True)
    downloaded_pathnames = access_module.download_price_data()
    for year in range(1995, 2022 + 1):
        ok = False
        for path in downloaded_pathnames:
            if str(year) in str(path):
                str1 = "\"" + str(year) + "-01-01 00:00"
                str2 = "\"" + str(year) + "-12-31 00:00"
                f1 = False
                f2 = False
                with open(path, 'r', newline='', encoding='utf-8') as file:
                    reader = csv.reader(file)
                    row_count = sum(1 for row in reader)
                    ch_count = max([len(row) for row in reader])
                    for row in reader:
                        if str1 in str(row):
                            f1 = True
                        if str2 in str(row):
                            f2 = True
                assert row_count <= 2000000
                assert ch_count <= 2048
                assert f1
                assert f2
                ok = True
        assert ok
    for path in downloaded_pathnames:
        os.remove(path)
    os.rmdir("tmp_data")

@pytest.mark.slow_locally
def test_download_postcode_data(access_module):
    pass
    os.makedirs("tmp_data", exist_ok=True)
    path = access_module.download_postcode_data()
    assert os.path.isfile(path)
    with open(path, 'r', newline='', encoding='utf-8') as file:
        reader = csv.reader(file)
        row_count = sum(1 for row in reader)
        ch_count = max([len(row) for row in reader])
    assert(row_count <= 5000000)
    assert ch_count <= 2048
    os.remove(path)    
    os.rmdir("tmp_data")

@pytest.mark.slow_locally
def test_joined_data(access_module):
    pass
    os.makedirs("tmp_data", exist_ok=True)
    downloaded_pathnames = access_module.download_price_data()
    postcode_path = access_module.download_postcode_data()
    joined_paths = access_module.join_all_tables(downloaded_pathnames, postcode_path)
    def change_key(e):
        return str(e)
    joined_paths = sorted(joined_paths, key=change_key)
    downloaded_pathnames = sorted(downloaded_pathnames, key=change_key)
    assert(len(joined_paths) == len(downloaded_pathnames))
    totlen = 0
    for i in range(len(joined_paths)):
        pj = joined_paths[i]
        po = downloaded_pathnames[i]
        dfj = pd.read_csv(Path(pj))['postcode'].sort_values()
        dfo = pd.read_csv(Path(po), header=0)[3].sort_values()
        assert(len(dfj) <= len(dfo))
        totlen += len(dfj)
        pj = 0
        po = 0
        nmatch = 0
        while (pj < len(dfj) and po < len(dfo)):
            if dfj.iloc[pj] == dfo.iloc[po]:
                nmatch += 1
                pj += 1
                po += 1
            else:
                assert ' ' in dfo.iloc[po]
                assert len(dfo[po]) >= 7
                po += 1
        assert nmatch * 100 >= len(dfo) * 95
    assert totlen == 28210620
    for path in downloaded_pathnames:
        os.remove(path)
    for path in joined_paths:
        os.remove(path)
    os.remove(postcode_path)    
    os.rmdir("tmp_data")