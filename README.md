# `AFFOGATO`
gAlactic Faint Feature extractiOn with GALFITM-bAsed Tools in pythOn

Used in [Nolan et al., 2025](https://iopscience.iop.org/article/10.3847/1538-4357/adec95) and submitted separately to JOSS. [The example file](example.ipynb) should work in an environment installed with `Jupyter` and `affogato-agn` - I use Conda, and add nb_conda_kernels to the environment setup and install `affogato-agn` separately with `pip`.

Note that the distribution name of the package is `adffogato-an` but the import name is `affogato`.  This does render it incompatible with the `affogato` package, so do not install it (it predates this package, so this is unavoidable).

## General Notes for Use:
These wrapper functions generally assume all of the `GALFITM` work is being done 
in a fairly clean directory, where files are not being saved permanently, 
instead being backed up somewhere else if needed, and other intermediate 
processes are discarded. I've tried to make these functions as general as is
practical, and noted in comments where the code becomes specific to my use 
case.

NOTA BENE: On my Mac, the `GALFITM` executable did not work until placing it in
the src folder of Anaconda3, and only then did it throw an error stating it was
being blocked by the secuity software of my Mac.  I was then able to override
that block in the Privacy & Security settings.  I provide options for where to
send your download, but your mileage may vary.
