# Creates tfrecords with prepared pretraining data for Masked Language Modelling 
# The tfrecords contain the fields:
#       input_ids with the token ids (some of them being masks)
#       lm_label_ids with the labels of the masks & -100 elsewhere
# We do not need attn_masks as we just fill up the whole seq_len so they can be created dynamically as a tensor of 1's


import argparse
import os
from pathlib import Path

import ftfy
import tensorflow as tf
from lm_dataformat import Reader # Could be internalized
from tqdm import tqdm
import logging
from multiprocessing import Pool, cpu_count
from itertools import repeat

from tokenization import Tokenizer, WWMTokenizer

parser = argparse.ArgumentParser()

parser.add_argument("--input_dir", type=str, help="Path to where your files are located")
parser.add_argument("--output_dir", type=str, default="./tfrecords", help="Where to put tfrecords")
parser.add_argument("--name", type=str, default="clue", help="Name of output files will be name_i.tfrecords where i is the number of the file")
parser.add_argument("--files_per", type=int, default=100000, help="Text files per tfrecord") # 1000 ~~ 20.8 MB 
parser.add_argument("--minimum_size", type=int, default=100, help="Minimum size a document has to be to be included")
parser.add_argument("--ftfy", action="store_true", help="normalize with ftfy - False by default")
parser.add_argument("--processes", type=int, default=0, help="Number of processes to use. Defaults to cpu count.")
parser.add_argument("--seq_len", type=int, default=2048, help="Sequence Len")
parser.add_argument("--cls_id", nargs="+", type=int, default=[101], help="CLassifier placed at the beg. of each example, defaults to BERTs [CLS] id")
parser.add_argument("--sep_id", nargs="+", type=int, default=[102], help="Seperator placed at the end of each example, defaults to BERTs [SEP] id")
parser.add_argument("--wwm", action="store_true", help="Whether to use Whole-Word Masking - False by default to save time")


args = parser.parse_args()


def _int64_feature(value):
    """
    Returns an int64_list from a bool / enum / int / uint.
    """
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def write_to_file(writer, input_ids, labels):
    """
    writes data to tfrecord file
    """
    feature = {
        "input_ids": _int64_feature(input_ids),
        "labels": _int64_feature(labels)
    }
    tf_example = tf.train.Example(features=tf.train.Features(feature=feature))
    writer.write(tf_example.SerializeToString())


def write_files(input_ids_list_array, labels_list_array, files_per, output_dir, out_name, start_no, write_remainder=False, process_no=None):
    # writes a list of files to .tfrecords
    if input_ids_list_array == None:
        return
    input_ids_list_array = split_list(input_ids_list_array, files_per)
    labels_list_array = split_list(labels_list_array, files_per)

    if len(input_ids_list_array[-1]) != files_per and not write_remainder: # pop the last file if it's length != files per
        input_ids_remainder = input_ids_list_array.pop(-1)
        labels_remainder = labels_list_array.pop(-1)
    else:
        input_ids_remainder, labels_remainder = None, None # assuming files = remainder from an old chunk here
        files_per = len(input_ids_list_array[-1])

    for input_ids_list, labels_list in zip(input_ids_list_array, labels_list_array):
        fp = f"{output_dir}/{out_name}_{start_no}"
        if process_no is not None:
            fp += f"_{process_no}"
        fp += f"_{files_per}" # add number of files in tfrecord to end of fp
        fp += ".tfrecords"
        with tf.io.TFRecordWriter(fp) as writer:
            for input_ids, labels in zip(input_ids_list, labels_list):
                write_to_file(writer, input_ids, labels)
        start_no += 1
    return start_no, input_ids_remainder, labels_remainder


def split_list(l, n):
    # splits list/string into n size chunks
    return [l[i:i+n] for i in range(0, len(l), n)]


def split_ids_labs(ids, labs, n):
    # splits list/string into n size chunks
    return [args.cls_id + ids[i:i+n-2] + args.sep_id for i in range(0, len(ids), n-2)], [[-100] + labs[i:i+n-2] + [-100] for i in range(0, len(labs), n-2)]


def archive_to_tokens(f, encoder, args):
    # Generator that yields the contents of the files in an archive
    # if data_to_prepend is not None, prepend data_to_prepend + a EOS separator to the encoded data
    reader = Reader(f)
    for doc in reader.stream_data(threaded=False):
        if args.ftfy: # fix text with ftfy if specified
            doc = ftfy.fix_text(doc, normalization='NFKC')

        input_ids, labels = encoder.tokenize_fast(doc)
        #doc = encoder.encode(doc) + args.separator # read document from lmd and append separator token

        yield split_ids_labs(input_ids, labels, args.seq_len) # split into n_ctx + 1 size chunks

def read_checkpoint(checkpoint_path, resume_from_checkpoint=True):
    # init checkpointing
    if resume_from_checkpoint and os.path.isfile(checkpoint_path):
        try:
            with open(checkpoint_path, "r") as f:
                resume_files_processed = f.read().split(", ")
            return resume_files_processed
        except:
            pass
    return []


def create_tfrecords(params, write_remainder=True, write_every_n_files=1, resume_from_checkpoint=True, display_pbar=True):
    # iterates through files in input_dir, splitting into <args.chunk_size> chunks and saving a tfrecords file every <args.files_per> chunks.
    files, args, process_no = params
    if args.wwm:
        print("WWM Masking ON")
        enc = WWMTokenizer(args.seq_len)
    else:
        print("No WWM Masking")
        enc = Tokenizer()

    # init metadata
    discarded_files = 0
    files_processed = 0
    tfrecord_count = 0
    pbar = tqdm(desc=f"Writing TFRecord Files to {args.output_dir}. Parsed 0 input files. files_written ", disable= not display_pbar)
    checkpoint_path = f"{args.output_dir}/processed_files.txt"

    input_ids_to_prepend = []
    labels_to_prepend = []

    input_ids_list_array = []
    labels_list_array = []

    files_processed_list = []

    for f in files:
        # Read in most updated list of processed files & skip if already processed
        resume_files_processed = read_checkpoint(checkpoint_path, resume_from_checkpoint)
        if f in resume_files_processed:
            continue
        for input_ids_list, labels_list in archive_to_tokens(f, enc, args): # input_ids_list is a whole file chunked in lists of seq_len
            files_processed += 1

            # if the last chunk < chunk size, but > minimum_size, take it and append it to the beginning of the next file
            n_tokens = len(input_ids_list[-1])
            if n_tokens < args.seq_len:
                input_ids_last = input_ids_list.pop(-1)
                labels_last = labels_list.pop(-1)
                if n_tokens >= args.minimum_size:
                    input_ids_to_prepend.extend(input_ids_last)
                    labels_to_prepend.extend(labels_last)
                else:
                    discarded_files += 1

            if len(input_ids_to_prepend) >= args.seq_len:
                # if length of data_to_prepend becomes greater than chunk size, add concatted files to tokenized files
                input_ids_list_array.append(input_ids_to_prepend[:args.seq_len])
                input_ids_to_prepend = input_ids_to_prepend[args.seq_len:]

                labels_list_array.append(labels_to_prepend[:args.seq_len])
                labels_to_prepend = labels_to_prepend[args.seq_len:]

            # add tokenized files > chunk size to main array
            input_ids_list_array.extend(input_ids_list)
            labels_list_array.extend(labels_list)

            if len(labels_list_array) >= args.files_per * write_every_n_files: # write every n files
                _tfrecord_count, input_ids_remainder, labels_remainder = write_files(input_ids_list_array, labels_list_array, files_per=args.files_per, output_dir=args.output_dir, out_name=args.name, start_no = tfrecord_count, process_no=process_no)
                pbar.update(_tfrecord_count - tfrecord_count) # update progress bar
                pbar.set_description(f"Writing TFRecord Files to {args.output_dir}. Parsed {files_processed} input files. files_written ")
                tfrecord_count = _tfrecord_count
                input_ids_list_array = input_ids_remainder if input_ids_remainder is not None else [] # add remaining files to next chunk
                labels_list_array = labels_remainder if labels_remainder is not None else []
                with open(f"{checkpoint_path}", "a") as myfile:
                    for x in files_processed_list:
                        myfile.write(f"{x}, ")
                    files_processed_list = []

        # Save the file names to skip next time if not doing all in one go
        files_processed_list.append(f)

    if len(labels_list_array) >= args.files_per: # also write at end
        _tfrecord_count, input_ids_remainder, labels_remainder = write_files(input_ids_list_array, labels_list_array, files_per=args.files_per, output_dir=args.output_dir, out_name=args.name, start_no=tfrecord_count, process_no=process_no)
        pbar.update(_tfrecord_count - tfrecord_count)
        pbar.set_description(f"Writing TFRecord Files to {args.output_dir}. Parsed {files_processed} input files. files_written ")
        tfrecord_count = _tfrecord_count
        with open(f"{checkpoint_path}", "a") as myfile:
            for x in files_processed_list:
                myfile.write(f"{x}, ")
            files_processed_list = []
    else:
        input_ids_remainder = input_ids_list_array # add remaining to remainder
        labels_remainder = labels_list_array 

    if write_remainder:
        # write out the remaining files even if there's less than files_per
        write_files(input_ids_list_array, labels_list_array, files_per=args.files_per, output_dir=args.output_dir, out_name=args.name, start_no=tfrecord_count, write_remainder=True)

    successful_files = files_processed - discarded_files
    return {"discarded": discarded_files, "processed": files_processed, "successful": successful_files}


def create_tfrecords_mp(files, args):
    files = split_list(files, len(files) // args.processes) # -(- ceils the rounding > 1 // 2 = 0; becomes 1 instead
    with Pool(processes=args.processes) as pool:
        pbar = tqdm(pool.imap(create_tfrecords, zip(files, repeat(args), range(len(files)))))
        meta = {"discarded": 0, "processed": 0, "successful": 0}
        for results in pbar:
            pbar.update()
            for k, v in results.items():
                meta[k] += v # update metadata
        return meta


def get_files(input_dir, filetypes=None):
    # gets all files of <filetypes> in input_dir
    if filetypes == None:
        filetypes = ["jsonl.zst", ".txt", ".xz", ".tar.gz", ".zip"]
    files = [list(Path(input_dir).glob(f"*{ft}")) for ft in filetypes]
    return [str(item) for sublist in files for item in sublist] # flatten list of list -> list and stringify Paths

if __name__ == "__main__":
    os.makedirs(args.output_dir, exist_ok=True) # make output dir if it doesn't exist
    files = get_files(args.input_dir)

    #args.chunk_size += 1 # we shift the data by 1 to the right for targets, so increment the chunk size here

    if args.processes == 0:
        args.processes = cpu_count()
    if args.processes > 1:
        results = create_tfrecords_mp(files, args)
    else:
        results = create_tfrecords((files, args, 0), display_pbar=True)
    print(results)