defmodule Learnx.MixProject do
  use Mix.Project

  @source_url "https://github.com/matiascr/learnx"
  @version "0.2.0"

  def project do
    [
      app: :sphinx,
      version: @version,
      elixir: "~> 1.14",
      start_permanent: Mix.env() == :prod,
      name: "Learnx",
      description: description(),
      package: package(),
      deps: deps()
    ]
  end

  # Run "mix help compile.app" to learn about applications.
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
      files: ~w(lib .formatter.exs mix.exs README* LICENSE* CHANGELOG* src),
      licenses: ["Apache-2.0"],
      links: %{"GitHub" => @source_url}
    ]
  end

  # Run "mix help deps" to learn about dependencies.
  defp deps do
    [
      # {:dep_from_hexpm, "~> 0.3.0"},
      # {:dep_from_git, git: "https://github.com/elixir-lang/my_dep.git", tag: "0.1.0"}
      {:nx, "~> 0.2"},
      {:dialyxir, "~> 1.0", only: [:dev], runtime: false}
    ]
  end
end
