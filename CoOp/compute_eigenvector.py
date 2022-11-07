import os
import numpy as np
from sklearn.decomposition import PCA
import torch
import argparse
parser = argparse.ArgumentParser(description='Subspace Prompt Tuning')
parser.add_argument('--ckpt_path', type=str, required=True)
parser.add_argument('--start', type=int, required=True)
parser.add_argument('--finish', type=int, required=True)
parser.add_argument('--save_name', type=str, required=True)
parser.add_argument('--n_components', type=int, required=True)
parser.add_argument('--solver', type=str, default="full")
args = parser.parse_args()

W = []
for i in range(args.start, args.finish+1):
    ckpt_name = os.path.join(args.ckpt_path, "model.pth.tar-{}".format(i))
    if os.path.exists(ckpt_name):
        ckpt = torch.load(ckpt_name)["state_dict"]["ctx"]
        ckpt = ckpt.detach().cpu().numpy().reshape(-1)
        W.append(ckpt)

W = np.array(W)
pca = PCA(n_components=args.n_components, svd_solver=args.solver)
pca.fit_transform(W)
U = np.array(pca.components_)
print ('ratio:', pca.explained_variance_ratio_)
print ("W.shape = {}, U.shape = {}".format(W.shape, U.shape))

U = torch.from_numpy(U)
torch.save(U, args.save_name)
print(args.save_name, '\n')
