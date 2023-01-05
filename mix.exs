defmodule Learnx.MixProject do
  use Mix.Project

  @source_url "https://github.com/matiascr/learnx"
  @version "0.2.0"

  def project do
    [
      app: :learnx,
      version: @version,
      elixir: "~> 1.14",
      start_permanent: Mix.env() == :prod,
      name: "Learnx",
      description: description(),
      package: package(),
      deps: deps(),
      docs: docs()
    ]
  end

  def application do
    [
      extra_applications: [:logger]
    ]
  end

  defp description do
    "Machine Learning algorithm implementations in pure elixir"
  end

  defp package do
    [
      maintainers: ["Matias Carlander-Reuterfelt"],
      files: ~w(lib .formatter.exs mix.exs README* LICENSE* ),
      licenses: ["Apache-2.0"],
      links: %{"GitHub" => @source_url}
    ]
  end

  defp deps do
    [
      {:ex_doc, ">= 0.29.0", only: :dev, runtime: false},
      {:nx, "~> 0.2"},
      {:dialyxir, "~> 1.0", only: [:dev], runtime: false}
    ]
  end

  defp docs do
    [
      main: "Learnx",
      groups_for_modules: [
        Regression: [
          Learnx.LinearRegression,
          Learnx.PolynomialRegression
        ]
      ]
    ]
  end
end
