import pickle

# 替换为你的 .pkl 文件路径
file_path = '/mnt/public/yhz/jiangzj/code/3d_understanding/BIP3D/submission.pkl'

# 打开文件并加载内容
with open(file_path, 'rb') as file:
    data = pickle.load(file)

# 打印或操作数据
print(data.keys())