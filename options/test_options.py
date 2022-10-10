from .base_options import BaseOptions


class TestOptions(BaseOptions):
    def initialize(self,  parser):
        parser = BaseOptions.initialize(self, parser)

        parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here')
        parser.add_argument('--sample_num', type=int, default=10, help='number of diverse results to sample')

        self.isTrain = False

        return parser
