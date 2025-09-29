
class TaskFlow:
    def __init__(self, args):
        self.args = args
        if args.task == "node_classification":
            from tasks.node_classification_task import NodeClassificationTask
            self.task = NodeClassificationTask(args)

    def run(self):
        self.task.run()

