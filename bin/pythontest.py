from liblda import *

a={1:
    ['/slfs1/users/hedi7/asr/CNNSV/data/feat_cmvn_plp/bkg_utt_lvl/features/cmvn/f001_01_001.plp',
    '/slfs1/users/hedi7/asr/CNNSV/data/feat_cmvn_plp/bkg_utt_lvl/features/cmvn/f001_01_002.plp',
    '/slfs1/users/hedi7/asr/CNNSV/data/feat_cmvn_plp/bkg_utt_lvl/features/cmvn/f001_01_003.plp',
    '/slfs1/users/hedi7/asr/CNNSV/data/feat_cmvn_plp/bkg_utt_lvl/features/cmvn/f001_01_004.plp',
    ],
    2:
    ['/slfs1/users/hedi7/asr/CNNSV/data/feat_cmvn_plp/dev_utt_lvl/features/cmvn/m100_09_021.plp',
        '/slfs1/users/hedi7/asr/CNNSV/data/feat_cmvn_plp/dev_utt_lvl/features/cmvn/m100_09_022.plp',
        '/slfs1/users/hedi7/asr/CNNSV/data/feat_cmvn_plp/dev_utt_lvl/features/cmvn/m100_09_023.plp']
    }

print fitlda(a)
