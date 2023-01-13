#coding=utf8
import sys, os, time
from torch.optim import Adam,AdamW
from torch.optim.lr_scheduler import StepLR
install_path = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(install_path)

from utils.args import init_args
from utils.initialization import *
from utils.example_bert import Example
from utils.batch import from_example_list
from utils.vocab import PAD
from model.slu_baseline_tagging import SLUTagging
from model.slu_bert_tagging import SLUTaggingBERT, SLUTaggingBERTCascaded,SLUTaggingBERTMultiHead

import logging

debug0=0 #whether to extend training dataset cais
debug1=0 #whether to extend training dataset ecdt
debug2=0 #whether use cascaded
debug3=1 #whether multihead
debug4=1 #whether augmentation 

# initialization params, output path, logger, random seed and torch.device
args = init_args(sys.argv[1:])
set_random_seed(args.seed)
device = set_torch_device(args.device)
print("Initialization finished ...")
print("Random seed is set to %d" % (args.seed))
print("Use GPU with index %s" % (args.device) if args.device >= 0 else "Use CPU as target torch device")

start_time = time.time()
if debug4:
    train_path = os.path.join(args.dataroot, 'train_augment.json')
else:
    train_path = os.path.join(args.dataroot, 'train.json')
if debug0:
    train_path_cais = os.path.join(args.dataroot, 'train_cais.json')
else:
    train_path_cais = None

if debug1:
    train_path_ecdt = os.path.join(args.dataroot, 'train_ecdt.json')
else:
    train_path_ecdt = None

dev_path = os.path.join(args.dataroot, 'development.json')
model_name=args.model_name
info=args.info

###set logger begin

# Set the logging level
logging.basicConfig(level=logging.INFO)
# Get the logger
logger = logging.getLogger(__name__)
# Get the current time
now = time.strftime("%m-%d %H:%M", time.gmtime())
# Define the file path
file_path = os.path.join("exp", now + model_name.split('/')[-1] + str(args.lr) + " info "+info+" .txt")

# Check if the directory exists
if not os.path.exists(os.path.dirname(file_path)):
    # Create the directory
    os.makedirs(os.path.dirname(file_path))

# Open the file in write mode
# Add a file handler to the logger
fh = logging.FileHandler(file_path)
logger.addHandler(fh)

###set logger end


logger.info("Use pretrained model: ",model_name)
Example.configuration(args.dataroot, asr=args.use_asr, train_path=train_path, word2vec_path=args.word2vec_path,tokenizer_name=model_name,extend_cais=debug0,extend_ecdt=debug1)
train_dataset = Example.load_dataset(train_path,train_path_cais,train_path_ecdt)
dev_dataset = Example.load_dataset(dev_path)
logger.info("Load dataset and database finished, cost %.4fs ..." % (time.time() - start_time))
logger.info("Dataset size: train -> %d ; dev -> %d" % (len(train_dataset), len(dev_dataset)))

args.vocab_size = Example.word_vocab.vocab_size
args.pad_idx = Example.word_vocab[PAD]
args.num_tags = Example.label_vocab.num_tags
args.num_acts = Example.label_vocab.num_acts
args.num_slots = Example.label_vocab.num_slots
args.tag_pad_idx = Example.label_vocab.convert_tag_to_idx(PAD)


# model = SLUTagging(args).to(device)
if debug2:
    model=SLUTaggingBERTCascaded(args).to(device)
elif debug3:
    model=SLUTaggingBERTMultiHead(args).to(device)
else:
    model=SLUTaggingBERT(args).to(device)
# Example.word2vec.load_embeddings(model.word_embed, Example.word_vocab, device=device)


def set_optimizer(model, args):
    params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    grouped_params = [{'params': list(set([p for n, p in params]))}]
    optimizer = AdamW(grouped_params, lr=args.lr)
    return optimizer

def set_scheduler(optimizer,args):
    scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    return scheduler

def decode(choice):
    assert choice in ['train', 'dev']
    model.eval()
    dataset = train_dataset if choice == 'train' else dev_dataset
    predictions, labels = [], []
    total_loss, count = 0, 0
    with torch.no_grad():
        for i in range(0, len(dataset), args.batch_size):
            cur_dataset = dataset[i: i + args.batch_size]
            current_batch = from_example_list(args, cur_dataset, device, train=True)
            pred, label, loss = model.decode(Example.label_vocab, current_batch)
            for j in range(len(current_batch)):
                if any([l.split('-')[-1] not in current_batch.utt[j] for l in pred[j]]):
                    logger.info(current_batch.utt[j], pred[j], label[j])
            predictions.extend(pred)
            labels.extend(label)
            total_loss += loss
            count += 1
        metrics = Example.evaluator.acc(predictions, labels)
    # torch.cuda.empty_cache() # if use, it will trigger GPU device problem
    # gc.collect()
    return metrics, total_loss / count


if not args.testing:
    num_training_steps = ((len(train_dataset) + args.batch_size - 1) // args.batch_size) * args.max_epoch
    logger.info('Total training steps: %d' % (num_training_steps))
    optimizer = set_optimizer(model, args)
    # scheduler = set_scheduler(optimizer,args)
    nsamples, best_result = len(train_dataset), {'dev_acc': 0., 'dev_f1': 0.}
    train_index, step_size = np.arange(nsamples), args.batch_size
    logger.info('Start training ......')
    for i in range(args.max_epoch):
        start_time = time.time()
        epoch_loss = 0
        epoch_sep_loss=0
        epoch_tag_loss=0
        np.random.shuffle(train_index)
        model.train()
        count = 0
        for j in range(0, nsamples, step_size):
            cur_dataset = [train_dataset[k] for k in train_index[j: j + step_size]]
            current_batch = from_example_list(args, cur_dataset, device, train=True)
            _, loss, sep_loss, tag_loss = model(current_batch)
            if (epoch_sep_loss): epoch_sep_loss += sep_loss.item()
            if (epoch_tag_loss): epoch_tag_loss += tag_loss.item()
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            count += 1
        # scheduler.step()
        logger.info('Training: \tEpoch: %d\tTime: %.4f\tTraining Loss: %.4f\tSep Loss: %.4f\tTag Loss: %.4f' % (i, time.time() - start_time, epoch_loss / count,epoch_sep_loss / count,epoch_tag_loss / count))
        # torch.cuda.empty_cache()
        # gc.collect()

        start_time = time.time()
        metrics, dev_loss = decode('dev')
        dev_acc, dev_fscore = metrics['acc'], metrics['fscore']
        logger.info('Evaluation: \tEpoch: %d\tTime: %.4f\tDev acc: %.2f\tDev fscore(p/r/f): (%.2f/%.2f/%.2f)' % (i, time.time() - start_time, dev_acc, dev_fscore['precision'], dev_fscore['recall'], dev_fscore['fscore']))
        if dev_acc > best_result['dev_acc']:
            best_result['dev_loss'], best_result['dev_acc'], best_result['dev_f1'], best_result['iter'] = dev_loss, dev_acc, dev_fscore, i
            torch.save({
                'epoch': i, 'model': model.state_dict(),
                'optim': optimizer.state_dict(),
            }, open('model.bin', 'wb'))
            logger.info('NEW BEST MODEL: \tEpoch: %d\tDev loss: %.4f\tDev acc: %.2f\tDev fscore(p/r/f): (%.2f/%.2f/%.2f)' % (i, dev_loss, dev_acc, dev_fscore['precision'], dev_fscore['recall'], dev_fscore['fscore']))

    logger.info('FINAL BEST RESULT: \tEpoch: %d\tDev loss: %.4f\tDev acc: %.4f\tDev fscore(p/r/f): (%.4f/%.4f/%.4f)' % (best_result['iter'], best_result['dev_loss'], best_result['dev_acc'], best_result['dev_f1']['precision'], best_result['dev_f1']['recall'], best_result['dev_f1']['fscore']))
else:
    start_time = time.time()
    metrics, dev_loss = decode('dev')
    dev_acc, dev_fscore = metrics['acc'], metrics['fscore']
    logger.info("Evaluation costs %.2fs ; Dev loss: %.4f\tDev acc: %.2f\tDev fscore(p/r/f): (%.2f/%.2f/%.2f)" % (time.time() - start_time, dev_loss, dev_acc, dev_fscore['precision'], dev_fscore['recall'], dev_fscore['fscore']))
