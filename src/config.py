# Global configuration for the project

MODEL_NAME = "hfl/chinese-roberta-wwm-ext"
MAX_LENGTH = 256
BATCH_SIZE = 2
GRAD_ACCUM_STEPS = 4
NUM_EPOCHS = 200
PATIENCE = 10
LEARNING_RATE = 2e-5

MAIN_CODES = [
    'I10.x00x032', 'I20.000', 'I20.800x007', 'I21.401', 'I50.900x018'
]
OTHER_CODES = [
    'E04.101', 'E04.102', 'E11.900', 'E14.900x001', 'E72.101', 'E78.500',
    'E87.600', 'I10.x00x023', 'I10.x00x024', 'I10.x00x027', 'I10.x00x028',
    'I10.x00x031', 'I10.x00x032', 'I20.000', 'I25.102', 'I25.103', 'I25.200',
    'I31.800x004', 'I38.x01', 'I48.x01', 'I48.x02', 'I49.100x001', 'I49.100x002',
    'I49.300x001', 'I49.300x002', 'I49.400x002', 'I49.400x003', 'I49.900',
    'I50.900x007', 'I50.900x008', 'I50.900x010', 'I50.900x014', 'I50.900x015',
    'I50.900x016', 'I50.900x018', 'I50.907', 'I63.900', 'I67.200x011',
    'I69.300x002', 'I70.203', 'I70.806', 'J18.900', 'J98.414', 'K76.000',
    'K76.807', 'N19.x00x002', 'N28.101', 'Q24.501', 'R42.x00x004',
    'R91.x00x003', 'Z54.000x033', 'Z95.501', 'Z98.800x612'
]

main2id = {code: idx for idx, code in enumerate(MAIN_CODES)}
other2id = {code: idx for idx, code in enumerate(OTHER_CODES)}
num_main = len(MAIN_CODES)
num_other = len(OTHER_CODES)
