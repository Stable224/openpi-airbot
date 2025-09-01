from dataloop import DataLoopClient
import os

DATALOOP_HOST = os.environ.get('DATALOOP_HOST', "192.168.215.80")
DATALOOP_USERNAME = os.environ.get('DATALOOP_USERNAME', 'admin')
DATALOOP_PASSWORD = os.environ.get('DATALOOP_PASSWORD', '123456')
SNAPSHOT_ID = os.environ.get('SNAPSHOT_ID', 'f855354e-ba64-43b8-b4b2-29535627fe85')

dataloop = DataLoopClient(
    endpoint=DATALOOP_HOST,
    username=DATALOOP_USERNAME,
    password=DATALOOP_PASSWORD
)

station1_dir = "/data/openpi/dataset/station1"

station2_dir = "/data/openpi/dataset/station2"


# 确保目录存在，如果不存在则创建
os.makedirs(station1_dir, exist_ok=True)

os.makedirs(station2_dir, exist_ok=True)

dataset_list = dataloop.snapshots.load_snapshots_samples(snapshot_id=SNAPSHOT_ID, sample_type="train")
for sample_id in dataset_list:
    response = dataloop.snapshots.load_samples_data(sample_id=sample_id)
    filename = f"{sample_id}.mcap"  # 默认名
    filepath = os.path.join(station1_dir, filename)
    with open(filepath, 'wb') as f:
        f.write(response.read())
    print(f"{sample_id} save success!")


