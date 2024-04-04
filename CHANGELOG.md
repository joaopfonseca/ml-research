# Changelog

## [Unreleased](https://github.com/joaopfonseca/ml-research/tree/HEAD)

[Full Changelog](https://github.com/joaopfonseca/ml-research/compare/v0.5.0...HEAD)

**Deprecated:**

- Format code with new version of ``black`` and fix errors coming from new ``sklearn`` version [\#66](https://github.com/joaopfonseca/ml-research/issues/66)

## [v0.5.0](https://github.com/joaopfonseca/ml-research/tree/v0.5.0) (2024-04-04)

[Full Changelog](https://github.com/joaopfonseca/ml-research/compare/v0.4.2...v0.5.0)

**Implemented features and enhancements:**

- Implement metrics for evaluating synthetic data [\#63](https://github.com/joaopfonseca/ml-research/issues/63)
- Set ``utils.parallel\_loop`` to track job completions instead of job starts [\#51](https://github.com/joaopfonseca/ml-research/issues/51)

**Fixed bugs:**

- Error when running ``make\_mean\_sem\_table`` \(AttributeError: 'DataFrame' object has no attribute 'map'\)  [\#61](https://github.com/joaopfonseca/ml-research/issues/61)
- Fix applymap `FutureWarning` [\#42](https://github.com/joaopfonseca/ml-research/issues/42)

**Documentation updates:**

- Add ``mlresearch.show\_versions\(\)`` functions for bug reports [\#56](https://github.com/joaopfonseca/ml-research/issues/56)
- Add issues template and fix broken link on PR template [\#49](https://github.com/joaopfonseca/ml-research/issues/49)
- Add Changelog generator [\#45](https://github.com/joaopfonseca/ml-research/issues/45)
- Add CONTRIBUTING file [\#44](https://github.com/joaopfonseca/ml-research/issues/44)

**Closed issues:**

- Add type hints to code and fix ``make code-analysis`` warnings [\#57](https://github.com/joaopfonseca/ml-research/issues/57)
- Add Python 3.12 support [\#53](https://github.com/joaopfonseca/ml-research/issues/53)
- Make package `rich` an optional dependency [\#47](https://github.com/joaopfonseca/ml-research/issues/47)
- Rename ``data\_augmentation`` submodule to ``synthetic\_data`` [\#32](https://github.com/joaopfonseca/ml-research/issues/32)

## [v0.4.2](https://github.com/joaopfonseca/ml-research/tree/v0.4.2) (2023-09-28)

[Full Changelog](https://github.com/joaopfonseca/ml-research/compare/v0.4.1...v0.4.2)

**Fixed bugs:**

- `mlresearch.preprocessing.PipelineEncoder` is not preserving feature order [\#43](https://github.com/joaopfonseca/ml-research/issues/43)
- Fix bugs and deprecation warnings arising from dependency updates [\#41](https://github.com/joaopfonseca/ml-research/issues/41)
- ``census`` dataset URL is broken [\#40](https://github.com/joaopfonseca/ml-research/issues/40)
- Replace ``df.append`` functions to comply with pandas 2.0 [\#39](https://github.com/joaopfonseca/ml-research/issues/39)

## [v0.4.1](https://github.com/joaopfonseca/ml-research/tree/v0.4.1) (2023-06-02)

[Full Changelog](https://github.com/joaopfonseca/ml-research/compare/v0.4.0...v0.4.1)

**Fixed bugs:**

- ``adult`` dataset contains trailing white spaces [\#36](https://github.com/joaopfonseca/ml-research/issues/36)
- The number of non-metric + metric features in ``mlresearch.datasets.Datasets.summarize\_datasets\(\)`` is not matching the total features [\#35](https://github.com/joaopfonseca/ml-research/issues/35)

**Closed issues:**

- Add ``check\_random\_states`` to ``\_\_init\_\_.py`` in utils submodule [\#37](https://github.com/joaopfonseca/ml-research/issues/37)

## [v0.4.0](https://github.com/joaopfonseca/ml-research/tree/v0.4.0) (2023-01-24)

[Full Changelog](https://github.com/joaopfonseca/ml-research/compare/v0.4a2...v0.4.0)

**Implemented features and enhancements:**

- Create function ``dataframe\_to\_image`` [\#28](https://github.com/joaopfonseca/ml-research/issues/28)
- Include a pipeline-compatible version for one-hot encoding [\#19](https://github.com/joaopfonseca/ml-research/issues/19)
- Add more dataset information in ``summarize\_datasets`` [\#18](https://github.com/joaopfonseca/ml-research/issues/18)
- ``sklearn.metrics.SCORERS`` is deprecated [\#16](https://github.com/joaopfonseca/ml-research/issues/16)
- Modify parameter names in ``make\_bold`` function to ensure parameter name consistency across utils function [\#15](https://github.com/joaopfonseca/ml-research/issues/15)
- Make pytorch models sklearn compatible [\#9](https://github.com/joaopfonseca/ml-research/issues/9)
- Move secondary dependencies to optional dependencies [\#8](https://github.com/joaopfonseca/ml-research/issues/8)

**Documentation updates:**

- Move LaTeX-related functions to their own submodule [\#29](https://github.com/joaopfonseca/ml-research/issues/29)
- Fix some of the errors/warnings in the documentation [\#22](https://github.com/joaopfonseca/ml-research/issues/22)
- Add missing docstring to GeometricSMOTE's ``\_encode\_categorical`` [\#17](https://github.com/joaopfonseca/ml-research/issues/17)

**Closed issues:**

- Fix deprecation warnings [\#21](https://github.com/joaopfonseca/ml-research/issues/21)
- Consider modifying default BYOL hyper-parameters for smaller batch sizes [\#11](https://github.com/joaopfonseca/ml-research/issues/11)
- Add Semi-supervised learning implementation [\#7](https://github.com/joaopfonseca/ml-research/issues/7)

## [v0.4a2](https://github.com/joaopfonseca/ml-research/tree/v0.4a2) (2022-10-10)

[Full Changelog](https://github.com/joaopfonseca/ml-research/compare/v0.4a1...v0.4a2)

**Closed issues:**

- Add ML-Research to conda [\#5](https://github.com/joaopfonseca/ml-research/issues/5)

## [v0.4a1](https://github.com/joaopfonseca/ml-research/tree/v0.4a1) (2022-04-14)

[Full Changelog](https://github.com/joaopfonseca/ml-research/compare/v0.3.4...v0.4a1)

**Implemented features and enhancements:**

- Add function to describe and imbalance datasets [\#10](https://github.com/joaopfonseca/ml-research/issues/10)
- Get to 80% code coverage [\#1](https://github.com/joaopfonseca/ml-research/issues/1)

**Documentation updates:**

- Update ``make`` commands [\#12](https://github.com/joaopfonseca/ml-research/issues/12)

**Closed issues:**

- Finish BYOL implementation and tests [\#6](https://github.com/joaopfonseca/ml-research/issues/6)
- Move all CI/CD to GitHub Actions [\#3](https://github.com/joaopfonseca/ml-research/issues/3)

## [v0.3.4](https://github.com/joaopfonseca/ml-research/tree/v0.3.4) (2021-12-28)

[Full Changelog](https://github.com/joaopfonseca/ml-research/compare/v0.3.3...v0.3.4)

## [v0.3.3](https://github.com/joaopfonseca/ml-research/tree/v0.3.3) (2021-10-07)

[Full Changelog](https://github.com/joaopfonseca/ml-research/compare/v0.3.2...v0.3.3)

## [v0.3.2](https://github.com/joaopfonseca/ml-research/tree/v0.3.2) (2021-09-03)

[Full Changelog](https://github.com/joaopfonseca/ml-research/compare/v0.3.1...v0.3.2)

## [v0.3.1](https://github.com/joaopfonseca/ml-research/tree/v0.3.1) (2021-07-26)

[Full Changelog](https://github.com/joaopfonseca/ml-research/compare/v0.3.0...v0.3.1)

## [v0.3.0](https://github.com/joaopfonseca/ml-research/tree/v0.3.0) (2021-07-22)

[Full Changelog](https://github.com/joaopfonseca/ml-research/compare/v0.2.1...v0.3.0)

## [v0.2.1](https://github.com/joaopfonseca/ml-research/tree/v0.2.1) (2021-07-09)

[Full Changelog](https://github.com/joaopfonseca/ml-research/compare/v0.2.0...v0.2.1)

## [v0.2.0](https://github.com/joaopfonseca/ml-research/tree/v0.2.0) (2021-07-07)

[Full Changelog](https://github.com/joaopfonseca/ml-research/compare/0.1.0...v0.2.0)

## [0.1.0](https://github.com/joaopfonseca/ml-research/tree/0.1.0) (2021-07-05)

[Full Changelog](https://github.com/joaopfonseca/ml-research/compare/c6e279cfeabf058b78f504ca4a3d7bf9dfecf8bf...0.1.0)



\* *This Changelog was automatically generated by [github_changelog_generator](https://github.com/github-changelog-generator/github-changelog-generator)*
