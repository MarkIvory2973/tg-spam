import click
import tasks.train
import tasks.result
import tasks.prompt

@click.group()
def cli():
    pass

@cli.command()
@click.option("--root", default="./data/", help="Folder contains datasets and checkpoints")
@click.option("--batch-size", default=32, help="Batch size of dataset")
@click.option("--learning-rate", default=0.001, help="Learning rate of Adam")
@click.option("--gamma", default=0.9, help="Gamma of ExponentialLR")
@click.option("--epochs", default=25, help="Total epochs of training")
def train(root, batch_size, learning_rate, gamma, epochs):
    tasks.train.run(root, batch_size, learning_rate, gamma, epochs)
    
@cli.command()
@click.option("--root", default="./data/", help="Folder contains datasets and checkpoints")
def result(root):
    tasks.result.run(root)
    
@cli.command()
@click.option("--root", default="./data/", help="Folder contains datasets and checkpoints")
@click.option("--epoch", default=5, help="Epoch of model to use")
@click.option("--input", help="Text to be classified")
def prompt(root, epoch, input):
    tasks.prompt.run(root, epoch, input)
    
if __name__ == "__main__":
    cli()