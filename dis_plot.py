import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils.options import args_parser
from utils.get_dataset import get_dataset


num_users = 100
print_num_users = 10
num_classes = 10


def get_distribution(dataset_train):
    min_size = 0
    min_require_size = 1
    K = num_classes
    y_train = np.array(dataset_train.targets)
    N = len(dataset_train)
    dict_users = {}

    idx_batch = None
    while min_size < min_require_size:
        idx_batch = [[] for _ in range(num_users)]
        for k in range(K):
            idx_k = np.where(y_train == k)[0]
            np.random.shuffle(idx_k)
            proportions = np.random.dirichlet(np.repeat(1.0, num_users))
            proportions = np.array(
                [p * (len(idx_j) < N / num_users) for p, idx_j in zip(proportions, idx_batch)])
            proportions = proportions / proportions.sum()
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
            min_size = min([len(idx_j) for idx_j in idx_batch])

    for j in range(num_users):
        # np.random.shuffle(idx_batch[j])
        dict_users[j] = idx_batch[j]

    return dict_users

args = args_parser()
dataset_train, _, _ = get_dataset(args)
dict_users = get_distribution(dataset_train)
array = np.zeros((print_num_users, num_classes))
for i in range(print_num_users):
    print(len(dict_users[i]))
    for j in dict_users[i]:
        array[i][dataset_train[j][1]] += 1

print(array)
array = array.reshape(print_num_users * num_classes)
# print(array)

# print(sum(array))
# # 示例数据

# 客户ID
x = np.arange(0, print_num_users)
x = np.tile(x, num_classes)
x = np.sort(x)
# # 类别
y = np.arange(0, num_classes)
y = np.tile(y, print_num_users)

# sizes = np.random.randint(1, 100, 100)  # 气泡大小
# print(len(sizes))

# 设置绘图风格
sns.set(style="whitegrid")

# 绘制气泡热图
sns.scatterplot(x=x, y=y, size=array, sizes=(1, 400))
plt.tick_params(labelsize=20)
plt.subplots_adjust(left=0.12,right=0.95,top=0.95,bottom=0.17)
plt.xticks(x)
# plt.yticks(y)

# plt.title('Data Distribution')
plt.xlabel('Client ID',fontdict={'fontsize':24})
plt.ylabel('Class ID',fontdict={'fontsize':24})
plt.legend([], [], frameon=False)
plt.savefig('./output/111/cifar10.pdf')
# plt.show()
