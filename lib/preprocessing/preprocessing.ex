defmodule Learnx.Preprocessing do
  import Nx

  @type tensor :: Nx.Tensor.t()

  @doc """
  Validate input data.

  ## Parameters
    - x : list or tensor of shape (n_samples, n_features) or (n_samples,).
      The input samples.

    - y : list or tensor of shape (n_samples,).
      The targets.
  """
  @spec validate_data(list | tensor, list | tensor) :: {tensor, tensor} | RuntimeError
  def validate_data(x, y) when not is_tensor(x) and not is_tensor(y),
    do: validate_data(tensor(x), tensor(y))

  def validate_data(x, y) when is_tensor(x) and is_tensor(y) do
    x =
      case shape(x) do
        {_samples, _features} -> x
        _ -> x |> reshape({:auto, 1})
      end

    y =
      case shape(y) do
        {_samples, _features} -> y |> squeeze()
        _ -> y
      end

    error_message =
      "Error: x and y must have same sample size, but currently they have x: #{axis_size(x, 0)} and y: #{axis_size(y, 0)}"

    case axis_size(x, 0) == axis_size(y, 0) do
      true -> {x, y}
      _ -> raise error_message
    end
  end
end
