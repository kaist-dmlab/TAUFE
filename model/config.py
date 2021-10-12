import math

mode = 'classification'
#mode = 'bbox'

if mode == 'classification':
    NUM_TRAIN = 50000  # N
    NUM_VAL = 50000 - NUM_TRAIN
    OOD_RATE = 1
    NUM_TRAIN_OOD = NUM_TRAIN * OOD_RATE  # N
    NUM_VAL_OOD = 50000 - NUM_TRAIN_OOD

    NUM_CLASS = 10
    SAMPLE_PER_CLASS = 500

    BATCH = 128 # B
    WEIGHT = 1  # lambda

    NUM_BATCH = math.ceil(NUM_CLASS*SAMPLE_PER_CLASS/BATCH)
    NUM_UPDATE = 5000 #??

    EPOCH = 200 #max(200, int(NUM_UPDATE/NUM_BATCH))
    EPOCH_DETACH = int(4*EPOCH/5) #int(3*EPOCH/4)+10 #EPOCH
    LR = 0.1
    MILESTONES = [int(EPOCH/2), int(3*EPOCH/4)]
    MOMENTUM = 0.9
    WDECAY = 5e-4

    print("Mode: {}, Weight: {}, Batch: {}, Epoch: {}, EPOCH_DETACH: {}, NUM_CLASS: {}, SAMPLE_PER_CLASS: {}"\
          .format(mode, WEIGHT, BATCH, EPOCH, EPOCH_DETACH, NUM_CLASS, SAMPLE_PER_CLASS))

elif mode == 'bbox':
    NUM_CLASS = 200
    SAMPLE_PER_CLASS = 10

    NUM_TRAIN = NUM_CLASS*SAMPLE_PER_CLASS
    OOD_RATE = 1
    NUM_TRAIN_OOD = NUM_TRAIN * OOD_RATE  # N

    BATCH = 32 # B
    WEIGHT = 1  # lambda

    NUM_BATCH = math.ceil(NUM_TRAIN/BATCH)
    NUM_UPDATE = 5000

    EPOCH = max(100, int(NUM_UPDATE/NUM_BATCH))
    EPOCH_DETACH = int(2*EPOCH/3) #EPOCH
    LR = 0.001
    MILESTONES = [int(EPOCH/2), int(3*EPOCH/4)]
    MOMENTUM = 0.9
    WDECAY = 5e-4

    print("Mode: {}, Weight: {}, Batch: {}, Epoch: {}".format(mode, WEIGHT, BATCH, EPOCH))