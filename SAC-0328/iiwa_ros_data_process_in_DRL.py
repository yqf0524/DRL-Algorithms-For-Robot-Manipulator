import numpy as np
import pandas as pd


class DataProcessing(object):
    def __init__(self, file_name):
        self.joint_index = ['J1', 'J2', 'J3', 'J4', 'J5', 'J6', 'J7']
        self.config_file = 'src/training_algorithms/scripts/data/config_' + file_name + '.xlsx'
        self.path_joint_space_file = 'src/training_algorithms/scripts/data/path_joint_space_' + file_name + '.xlsx'
        self.path_cartesian_space_file = 'src/training_algorithms/scripts/data/path_cartesian_space_' + file_name + '.xlsx'
        self.file_name = file_name

        self.config_writer = pd.ExcelWriter(self.config_file)
        self.config_df = pd.DataFrame()
        self.config_num = 1

        self.path_joint_space_writer = pd.ExcelWriter(self.path_joint_space_file)
        self.path_joint_space_df =  pd.DataFrame()
        self.path_joint_space_sheet_num = 1

        self.path_cartesian_space_writer = pd.ExcelWriter(self.path_cartesian_space_file)
        self.path_cartesian_space_df =  pd.DataFrame()
        self.path_cartesian_space_sheet_num = 1

    def add_path_joint_space(self, config, step):
        self.path_joint_space_df[step] = pd.Series(config, index=self.joint_index)

    def add_config_df_by_col(self, config):
        self.config_df[self.config_num] = pd.Series(config, index=self.joint_index)
        self.config_num += 1

    def read_config_df_by_row(self, row_index):
        data = self.config_df.loc[row_index]
        return data.values
    
    def read_config_df_by_col(self, col_num):
        col = self.config_df.columns[col_num]
        return self.config_df[col].values
 
    def read_config_excel_by_row(self, row_index):
        data = self.read_config_excel.loc[row_index]
        return data.values
   
    def read_config_excel_by_col(self, col_num):
        df = self.read_config_excel()
        col = df.columns[col_num]
        return df[col].values

    def read_config_excel(self):
        print('Reading ' + self.file_name + ' configurations ...')
        data = pd.read_excel(self.config_file, header=0, index_col=0, engine='openpyxl')
        self.config_num = data.columns[-1]
        return data

    def read_path_joint_space_excel(self, sheet_name):
        print('Reading ' + self.file_name + ' ' + sheet_name + ' in joint space ...')
        data = pd.read_excel(self.path_joint_space_file, sheet_name=sheet_name, 
                             header=0, index_col=0, engine='openpyxl')
        return data
    
    def read_path_cartesian_space_excel(self, sheet_name):
        print('Reading ' + self.file_name + ' ' + sheet_name + ' in joint space ...')
        data = pd.read_excel(self.path_cartesian_space_file, sheet_name=sheet_name, 
                             header=0, index_col=0, engine='openpyxl')
        return data

    def save_config(self):
        print('Saving ' + self.file_name + ' configurations ...')
        self.config_df.to_excel(self.config_writer)
        self.config_writer.save()
        self.config_writer.close()

    def save_path_joint_space(self):
        print('Saving ' + self.file_name + ' path in joint space ...')
        sheet_name = 'path_' + str(self.path_joint_space_sheet_num)
        self.path_joint_space_df.to_excel(self.path_joint_space_writer, sheet_name=sheet_name)
        self.path_joint_space_writer.save()
        self.path_joint_space_sheet_num += 1

    def save_path_cartesian_space(self, path_xyz, path_rpy):
        print('Saving ' + self.file_name + ' path in cartesian space ...')
        sheet_name = 'path_' + str(self.path_cartesian_space_sheet_num)
        self.path_cartesian_space_df['x'] = pd.Series(path_xyz[0])
        self.path_cartesian_space_df['y'] = pd.Series(path_xyz[1])
        self.path_cartesian_space_df['z'] = pd.Series(path_xyz[2])
        self.path_cartesian_space_df['a'] = pd.Series(path_rpy[0])
        self.path_cartesian_space_df['b'] = pd.Series(path_rpy[1])
        self.path_cartesian_space_df['c'] = pd.Series(path_rpy[2])
        self.path_cartesian_space_df.to_excel(self.path_cartesian_space_writer, sheet_name=sheet_name)
        self.path_cartesian_space_writer.save()
        self.path_cartesian_space_sheet_num += 1

