
import os
import glob
import h5py
import torch
import pandas as pd
from torch.utils.data import Dataset
class MILDataset(Dataset):

    def __init__(self, args,split, feat_dir, label_xlsx, genomics_txt,coords_dir):
        self.slide_ids = [item for item in split]

        self.feat_dir = feat_dir
        self.feat_files = self.get_feat()

        self.label_df = pd.read_csv(label_xlsx)
        self.label_df.set_index('slide_id', inplace=True)


        self.gene_df = pd.read_csv(genomics_txt, index_col=0).T
        # ==================================================
        self.args = args
        self.coords_dir = coords_dir
        self.coords_files = self.get_coords_files()
    def get_feat(self):
        feat_files = {}
        for slide_id in self.slide_ids:
            feat_paths = glob.glob(os.path.join(self.feat_dir, str(slide_id) + '.pt*'))
            slide_feats = []
            for feat_path in feat_paths:
                slide_feats.append(feat_path)
            feat_files[slide_id] = slide_feats
        return feat_files

    def get_coords_files(self):
        coords_files = {}
        for slide_id in self.slide_ids:

            h5_path = os.path.join(self.coords_dir, str(slide_id) + '.h5')
            if os.path.exists(h5_path):
                coords_files[slide_id] = h5_path
            else:

                coords_files[slide_id] = None
        return coords_files

    def get_others(self, slide_name, df):





        slide_name = slide_name



        time = df.loc[slide_name, 'survival_days']
        event = df.loc[slide_name, 'censorship']


        return time,event

    def __getitem__(self, idx):
        slide_name = self.slide_ids[idx]

        time,event = self.get_others(slide_name, self.label_df)

        feat_files = self.feat_files[slide_name]
        feats = torch.Tensor()
        for feat_file in feat_files:
            feat = torch.load(feat_file, map_location='cpu', weights_only=True)
            try:
                feat = torch.from_numpy(feat)
            except:
                pass
            feats = torch.cat((feats, feat), dim=0)


        patient_id = slide_name[:15]


        if patient_id in self.gene_df.index:

            gene_array = self.gene_df.loc[patient_id].values.astype('float32')
            gene_feat = torch.from_numpy(gene_array)
        else:

            print(f"当前尝试匹配的 ID: {patient_id}")
            print(f"索引示例: {list(self.gene_df.index[:3])}")
            raise KeyError(f"在基因 CSV 中找不到对应的病人 ID: {patient_id}，请检查匹配关系！")
        # ==================================================

        coords_path = self.coords_files[slide_name]
        coords = torch.Tensor()
        if coords_path:
            with h5py.File(coords_path, 'r') as hf:

                coords_dataset = hf['coords']
                coords_data = coords_dataset[:]
                coords = torch.from_numpy(coords_data).float()


        sample = {
            'slide_id': slide_name,
            'feat': feats,
            'time': time,
            'event': event,
            'gene': gene_feat,
            'coords': coords
        }
        return sample

    def __len__(self):
        return len(self.slide_ids)


class OtherMILDataset(Dataset):

    def __init__(self, args, split, feat_dir, coords_dir):
        self.slide_ids = [item for item in split]

        self.feat_dir = feat_dir
        self.feat_files = self.get_feat()




        self.args = args
        self.coords_dir = coords_dir
        self.coords_files = self.get_coords_files()

    def get_feat(self):
        feat_files = {}
        for slide_id in self.slide_ids:
            feat_paths = glob.glob(os.path.join(self.feat_dir, str(slide_id) + '.pt*'))
            slide_feats = []
            for feat_path in feat_paths:
                slide_feats.append(feat_path)
            feat_files[slide_id] = slide_feats
        return feat_files

    def get_coords_files(self):
        coords_files = {}
        for slide_id in self.slide_ids:

            h5_path = os.path.join(self.coords_dir, str(slide_id) + '.h5')
            if os.path.exists(h5_path):
                coords_files[slide_id] = h5_path
            else:

                print(f"ERROR -> {h5_path}")
                coords_files[slide_id] = None
        return coords_files

    def get_others(self, slide_name, df):

        slide_name = slide_name

        time = df.loc[slide_name, 'survival_days']
        event = df.loc[slide_name, 'censorship']

        return time, event

    def __getitem__(self, idx):
        slide_name = self.slide_ids[idx]



        feat_files = self.feat_files[slide_name]
        feats = torch.Tensor()
        for feat_file in feat_files:
            feat = torch.load(feat_file, map_location='cpu', weights_only=True)
            try:
                feat = torch.from_numpy(feat)
            except:
                pass
            feats = torch.cat((feats, feat), dim=0)



        coords_path = self.coords_files[slide_name]
        coords = torch.Tensor()
        if coords_path:
            with h5py.File(coords_path, 'r') as hf:

                coords_dataset = hf['coords']
                coords_data = coords_dataset[:]
                coords = torch.from_numpy(coords_data).float()


        sample = {
            'slide_id': slide_name,
            'feat': feats,
            'coords': coords
        }
        return sample

    def __len__(self):
        return len(self.slide_ids)






