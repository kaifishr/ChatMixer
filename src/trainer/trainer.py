import pathlib
import datetime
import os
import time

import torch
from torch.utils.tensorboard import SummaryWriter

from src.config.config import Config
from src.utils.stats import comp_stats_classification
from src.summary.summary import (
    add_graph,
    add_token_embedding_weights,
    add_position_embedding_weights,
    add_linear_weights,
)


class Trainer:
    """Trainer class.

    Attributes:
        model: PyTorch model.
        dataloader: Tuple holding training and test dataloader.
        config: Class holding configuration.

    Typical usage example:
        model = Model()
        dataloader = Dataloader()
        config = Config()
        trainer = Trainer(model, dataloader, config):
        trainer.run()
    """

    def __init__(self, model: torch.nn.Module, dataloader: tuple, config: Config) -> None:
        """Initializes Trainer."""
        self.model = model
        self.dataloader = dataloader
        self.config = config

        self.num_update_steps = config.trainer.num_update_steps

        runs_dir = config.dirs.runs
        dataset = config.dataloader.dataset

        uid = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S.%f")
        tag = config.tag

        log_dir = os.path.join(runs_dir, f"{uid}_{dataset}{f'_{tag}' if tag else ''}")

        self.writer = SummaryWriter(log_dir=log_dir)

        # Save config file
        file_path = pathlib.Path(self.writer.log_dir) / "config.txt"
        with open(file_path, "w") as file:
            file.write(self.config.__str__())

        train_loader, test_loader = dataloader

        # Add graph of model to Tensorboard.
        if config.summary.add_graph:
            add_graph(model=model, dataloader=train_loader, writer=self.writer, config=config)

        learning_rate = config.trainer.learning_rate
        weight_decay = config.trainer.weight_decay
        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )

        self.criterion = torch.nn.CrossEntropyLoss()

    def run(self):
        """Main training logic."""

        config = self.config
        writer = self.writer
        model = self.model
        optimizer = self.optimizer
        criterion = self.criterion
        device = config.trainer.device

        train_loader, test_loader = self.dataloader

        update_step = 0

        while update_step < self.num_update_steps:

            running_loss = 0.0
            running_accuracy = 0.0
            running_counter = 0

            model.train()
            t0 = time.time()

            for x_data, y_data in train_loader:

                # Get the inputs and labels.
                inputs, labels = x_data.to(device), y_data.to(device)

                # Zero the parameter gradients.
                optimizer.zero_grad(set_to_none=True)

                # Feedforward.
                outputs = model(inputs)

                outputs = outputs.view(-1, outputs.size(-1))
                labels = labels.view(-1)

                # Compute loss.
                loss = criterion(outputs, labels)

                # Backpropagation
                loss.backward()

                # Clip gradients
                if config.trainer.gradient_clipping.is_activated:
                    max_norm = config.trainer.gradient_clipping.max_norm
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

                # Gradient descent
                optimizer.step()

                # keeping track of statistics
                running_loss += loss.item()
                running_accuracy += (torch.argmax(outputs, dim=1) == labels).float().sum()
                running_counter += labels.size(0)

                if config.summary.save_train_stats.every_n_updates > 0:
                    if (update_step + 1) % config.summary.save_train_stats.every_n_updates == 0:
                        train_loss = running_loss / running_counter
                        train_accuracy = running_accuracy / running_counter

                        writer.add_scalar("train_loss", train_loss, global_step=update_step)
                        writer.add_scalar("train_accuracy", train_accuracy, global_step=update_step)

                        time_per_update = (time.time() - t0) / running_counter
                        writer.add_scalar(
                            "time_per_update", time_per_update, global_step=update_step
                        )

                        running_loss = 0.0
                        running_accuracy = 0.0
                        running_counter = 0

                        t0 = time.time()

                        print(f"{update_step:09d} {train_loss:.5f} {train_accuracy:.4f}")

                if config.summary.save_test_stats.every_n_updates > 0:
                    if (update_step + 1) % config.summary.save_test_stats.every_n_epochs == 0:
                        test_loss, test_accuracy = comp_stats_classification(
                            model=model,
                            criterion=criterion,
                            data_loader=test_loader,
                            device=device,
                        )
                        writer.add_scalar("test_loss", test_loss, global_step=update_step)
                        writer.add_scalar("test_accuracy", test_accuracy, global_step=update_step)

                self._write_summary(model=model, writer=writer, update_step=update_step)
                update_step += 1

        num_update_steps = config.trainer.num_update_steps
        self._write_summary(model=model, writer=writer, update_step=num_update_steps)
        writer.close()

    def _write_summary(self, model, writer, update_step: int) -> None:
        """"""
        config = self.config

        if config.summary.save_model.every_n_updates > 0:
            if update_step % config.summary.save_model.every_n_updates == 0:
                dataset = config.dataloader.dataset
                tag = f"_{config.tag}" if config.tag else ""
                model_name = f"{dataset}{tag}.pth"
                model_path = os.path.join(config.dirs.weights, model_name)
                torch.save(model.state_dict(), model_path)

        if config.summary.add_linear_weights.every_n_updates > 0:
            if update_step % config.summary.add_linear_weights.every_n_updates == 0:
                add_linear_weights(model=model, writer=writer, global_step=update_step)

        if config.summary.add_token_embeddings.every_n_updates > 0:
            if update_step % config.summary.add_token_embeddings.every_n_updates == 0:
                add_token_embedding_weights(model=model, writer=writer, global_step=update_step)

        if config.summary.add_position_embeddings.every_n_updates > 0:
            if update_step % config.summary.add_position_embeddings.every_n_updates == 0:
                add_position_embedding_weights(model=model, writer=writer, global_step=update_step)
