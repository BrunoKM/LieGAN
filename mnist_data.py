import jax
import jax.numpy as jnp
# from chex import PRNGKey
import tensorflow_datasets as tfds
from clu import deterministic_data
from clu import preprocess_spec


from dataclasses import dataclass
from preprocess import all_ops
# from augmented_dsprites import construct_augmented_dsprites


@dataclass
class DataConfig:
    angle: float
    batch_size: int
    dataset: str = "MNIST"
    aug_dsprites: None = None  # TODO
    train_split: str = "train[10000:60000]"
    val_split: str = "train[:10000]"
    test_split: None = None  # TODO
    shuffle_buffer_size: int = 50000

    @property
    def pp_train(self) -> str:
        return f'value_range(-1, 1)|random_rotate(-{self.angle}, {self.angle}, fill_value=-1)|keep(["image", "label"])'
    @property
    def pp_eval(self) -> str:
        return f'value_range(-1, 1)|random_rotate(-{self.angle}, {self.angle}, fill_value=-1)|keep(["image", "label"])'
    @property
    def num_val_examples(self) -> int:
        return int(self.val_split.split(":")[1].split("]")[0])


def get_data(
    config: DataConfig,
    rng,
):
    train_rng, val_rng, test_rng = jax.random.split(rng, 3)

    if config.dataset == "aug_dsprites":
        raise NotImplemented
        dataset = construct_augmented_dsprites(
            aug_dsprites_config=config.aug_dsprites,
            sampler_rng=train_rng,
        )
        dataset_or_builder = dataset
    else:
        dataset_builder = tfds.builder(config.dataset)
        dataset_builder.download_and_prepare()
        dataset_or_builder = dataset_builder

    local_batch_size = config.batch_size // jax.device_count()

    train_ds = deterministic_data.create_dataset(
        dataset_or_builder,
        split=tfds.split_for_jax_process(config.train_split),
        # This RNG key will be used to derive all randomness in shuffling, data
        # preprocessing etc.
        rng=train_rng,
        shuffle_buffer_size=config.shuffle_buffer_size,
        # Depending on TPU/other runtime, local device count will be 8/1.
        batch_dims=[jax.local_device_count(), local_batch_size],
        repeat_after_batching=False,
        preprocess_fn=preprocess_spec.parse(
            spec=config.pp_train,
            available_ops=all_ops(),
        ),
        shuffle="loaded",
    )

    if config.dataset == "aug_dsprites":
        raise NotImplemented
        dataset = construct_augmented_dsprites(
            aug_dsprites_config=config.aug_dsprites,
            sampler_rng=val_rng,  # Use a different RNG key for validation.
        )
        num_val_examples = config.num_val_examples
        dataset = dataset.take(num_val_examples)
        dataset_or_builder = dataset
        # Need to specify cardinality for the dataset manually
        cardinality = num_val_examples
    else:
        num_val_examples = dataset_builder.info.splits[config.val_split].num_examples
        cardinality = None
    # Compute how many batches we need to contain the entire val set.
    pad_up_to_batches = int(jnp.ceil(num_val_examples / config.batch_size))

    val_ds = deterministic_data.create_dataset(
        dataset_or_builder,
        split=tfds.split_for_jax_process(config.val_split),
        rng=val_rng,
        batch_dims=[jax.local_device_count(), local_batch_size],
        num_epochs=1,
        preprocess_fn=preprocess_spec.parse(
            spec=config.pp_eval,
            available_ops=all_ops(),
        ),
        # Pad with masked examples instead of dropping incomplete final batch.
        pad_up_to_batches=pad_up_to_batches,
        cardinality=cardinality,
        shuffle=False,
    )

    test_split = config.test_split
    if test_split is None:
        return train_ds, val_ds, None

    num_test_examples = dataset_builder.info.splits[test_split].num_examples
    # Compute how many batches we need to contain the entire test set.
    pad_up_to_batches = int(jnp.ceil(num_test_examples / config.batch_size))

    test_ds = deterministic_data.create_dataset(
        dataset_builder,
        split=tfds.split_for_jax_process(test_split),
        rng=test_rng,
        batch_dims=[jax.local_device_count(), local_batch_size],
        num_epochs=1,
        preprocess_fn=preprocess_spec.parse(
            spec=config.pp_eval,
            available_ops=all_ops(),
        ),
        # Pad with masked examples instead of dropping incomplete final batch.
        pad_up_to_batches=pad_up_to_batches,
        shuffle=False,
    )

    return train_ds, val_ds, test_ds
