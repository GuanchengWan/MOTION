from args import args
from utils.taskflow import TaskFlow
from utils.set_seed import set_random_seed


if args.seed is not None:
    set_random_seed(args.seed)
task = TaskFlow(args)
task.run()
