defmodule Learnx.TestHelper do
  import Nx

  def approx(num, _) when is_integer(num), do: num

  def approx(list, precision) when is_list(list),
    do: list |> Enum.map(&round(&1, precision))

  def approx(tensor, precision) when is_tensor(tensor) do
    case shape(tensor) do
      {} -> tensor |> to_number() |> round(precision)
      {_} -> tensor |> to_flat_list() |> approx(precision)
    end
  end

  def round(num, _) when is_integer(num), do: num
  def round(num, precision) when is_float(num), do: Float.round(num, precision)
end
