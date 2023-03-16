class Path(object):
    @staticmethod
    def db_dir(database):
        if database == 'ucf101':
            # folder that contains class labels
            root_dir = 'UCF-101'
            # Save preprocess data into output_dir
            output_dir = 'ucf101_results'

            return root_dir, output_dir
        elif database == 'hmdb51':
            # folder that contains class labels
            root_dir = 'hmdb51'

            output_dir = 'hmdb51_results'

            return root_dir, output_dir
        else:
            print('Database {} not available.'.format(database))
            raise NotImplementedError

    @staticmethod
    def model_dir():
        return '/hy-tmp/data/c3d-pretrained.pth'
