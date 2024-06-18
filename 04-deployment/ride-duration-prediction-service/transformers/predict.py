import pickle
import pandas as pd

if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test


@transformer
def transform(data, *args, **kwargs):
    """
    Template code for a transformer block.

    Add more parameters to this function if this block has multiple parent blocks.
    There should be one parameter for each output variable from each parent block.

    Args:
        data: The output from the upstream parent block
        args: The output from any additional upstream blocks (if applicable)

    Returns:
        Anything (e.g. data frame, dictionary, array, int, str, etc.)
    """
    year = int(kwargs.get('year'))
    month = int(kwargs.get('month'))

    with open('model.bin', 'rb') as f_in:
        _, model = pickle.load(f_in)

    y_pred = model.predict(data)

    y_pred = pd.Series(y_pred, name='y_pred').reset_index()

    y_pred['ride_id'] = f'{year:04d}/{month:02d}_' + y_pred.index.astype('str')

    return y_pred[['ride_id', 'y_pred']]


@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert output is not None, 'The output is undefined'
