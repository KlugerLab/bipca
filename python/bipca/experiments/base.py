from pathlib import Path
from shutil import rmtree, copyfile
from typing import Dict, TypedDict, Union, Optional
from urllib.parse import urlparse
import tasklogger
import numpy as np
from anndata import AnnData, read_h5ad

from ..base import ABC, abstractclassattribute, abstractmethod, classproperty
from .utils import download_url
from .exceptions import handle_multiple_exceptions


class DataFilters(TypedDict):
    """DataFilters: Dimensionally defined filters for data.

    DataFilters['obs'] is a dictionary with keys that match columns of a dataset's
    annotated AnnData.obs.

    Datafilters['var'] is a similar dictionary with keys into AnnData.var.
    """

    obs: Dict
    var: Dict


class Dataset(ABC):
    """Dataset: Abstract base class for Dataset objects."""

    _citation: str = abstractclassattribute()  # a bibtex citation
    _raw_urls: Dict[
        str, str
    ] = abstractclassattribute()  # Dictionary of local filenames to urls
    _filtered_url: str = (
        abstractclassattribute()
    )  # URL location of final filtered data. Leave blank if unavailable
    _filters: DataFilters = (
        abstractclassattribute()
    )  # A datafilters object that describes dataset filters. Built through mixins.

    # Abstract methods - these must be set by the subclass.
    @classmethod
    @abstractmethod
    def _annotate(cls, adata=None) -> AnnData:
        """_annotate: Annotate the raw data contained in `adata`.
        To be implemented in implementing classes of Dataset, either through
        multiple inheritance or direct descendants.
        If it's not implemented, this raises
        TypeError: Can't instantiate abstract class {cls} with abstract method
        filter"""
        return adata

    @abstractmethod
    def _process_raw_data(self) -> AnnData:
        """_process_raw_data: Process the raw data from whatever format it is
        downloaded from into an adata. MUST RETURN ADATA
        To be implemented by the final implementing dataset.
        """
        pass

    def __init__(
        self,
        base_data_directory: str = "/bipca_data",
        store_filtered_data: bool = True,
        store_raw_data: bool = False,
        store_raw_files: bool = False,
        logger: Optional[tasklogger.TaskLogger] = None,
        verbose: int = 1,
    ):
        self.verbose = verbose
        if logger is None:
            self.logger = tasklogger.TaskLogger(
                name=self.__class__.__name__, level=self.verbose, if_exists="increment"
            )

        self._base_data_directory = Path(base_data_directory).resolve()

        self.store_filtered_data = store_filtered_data
        self.store_raw_data = store_raw_data
        self.store_raw_files = store_raw_files
        self.__check_filters__()

    @classmethod
    def __check_filters__(cls, tmp_filters=None):
        if tmp_filters is None:
            tmp_filters = cls.filters
        for dimension_key, dimension_value in tmp_filters.items():
            for k, v in dimension_value.items():
                if isinstance(v, Dict):
                    if all((x not in v for x in ["min", "max"])):
                        raise KeyError(
                            f"Filter "
                            f"{cls.__name__}._filters"
                            f"['{dimension_key}']['{k}']"
                            " did not specify 'min' or 'max'."
                        )
                    else:
                        pass
                else:
                    raise TypeError(
                        "Filter "
                        f"{cls.__name__}._filters"
                        f"['{dimension_key}']['{k}']"
                        " was not a dict."
                    )
        return True

    # Properties
    @property
    def base_data_directory(self) -> Path:
        """base_data_directory: Root folder for bipca dataset storage.

        This folder is used to construct other paths for Dataset objects.

        Returns
        -------
        Path
            Root folder where `datasets` will be stored.
        """
        return self._base_data_directory

    @base_data_directory.setter
    def base_data_directory(self, val: Union[str, Path]):
        self._base_data_directory = Path(val).resolve()

    @property
    def dataset_directory(self) -> Path:
        """dataset_directory: Path to dataset directory.
        This directory will be used to store raw or filtered data files.

        Built from `instance.base_data_directory`, `instance.modality`, and
        `instance.__class__.__name__`.

        Returns
        -------
        Path
            Path to base folder for dataset.
        """
        return (
            self.base_data_directory
            / "datasets"
            / self.modality
            / self.__class__.__name__
        )

    @property
    def filtered_data_path(self) -> Path:
        """filtered_data_path: Path to filtered data `.h5ad` file.

        The path for the filtered file is `instance.dataset_directory/filtered.h5ad`

        Returns
        -------
        Path
            Path to filtered `.h5ad` file for dataset.
        """
        return self.dataset_directory / "filtered.h5ad"

    @property
    def raw_data_path(self) -> Path:
        """raw_data_path: Path to raw data `.h5ad` file.

        The path for the filtered file is `instance.dataset_directory/raw.h5ad`

        Returns
        -------
        Path
            Path to raw `.h5ad` file for dataset.
        """
        return self.dataset_directory / "raw.h5ad"

    @property
    def raw_files_directory(self) -> Path:
        """raw_files_directory: Path to raw files directory.

        Raw files are stored in `instance.dataset_directory/raw/`.

        Returns
        -------
        Path
            Path to raw files.
        """
        return self.dataset_directory / "raw"

    @property
    def raw_files_paths(self) -> Dict[str, Path]:
        """raw_files_paths: Paths to raw files.

        Raw files are stored in `instance.dataset_directory/raw/`.

        Returns
        -------
        Dict[str, Path]
            Filenames in `cls._raw_urls` to completed paths.
        """
        return {f: self.raw_files_directory / f for f in self._raw_urls.keys()}

    @classproperty
    def citation(cls) -> str:
        """citation: BibTeX citation entry for dataset.

        Defined by `_citation` in implementing subclasses.

        Returns
        -------
        str
        """
        return cls._citation

    @classproperty
    def raw_urls(cls) -> Dict[str, str]:
        """raw_urls: Filenames and URLs of raw data.

        `raw_urls` uses a dictionary to describe where to download raw data.
        The keys of the dictionary are local file names in
        `instance.raw_files_directory`,
        and the values of the dictionary are URLs to download from.

        Defined by `_raw_urls` in implementing subclasses.


        Returns
        -------
        Dict[str, str]
            _description_
        """
        return cls._raw_urls

    @classproperty
    def filtered_url(cls) -> str:
        """filtered_url: URL to filtered data.

        Defined by `_filtered_url` in implementing subclasses.

        Returns
        -------
        str

        """
        return cls._filtered_url

    @classproperty
    def filters(cls) -> Dict:
        """filters: Filters associated with dataset.

        Returns a filter dictionary, with keys 'obs' and 'var', that correspond to
        dictionaries of dimensional filters for adata objects.

        Each dimensional filter, e.g., cls.filters['obs'], must be a dictionary with
        keys that map to the annotated implementing class's AnnData. That is, if
        `cls.filters['obs']['foo']` is defined, then `cls.annotate(adata).obs['foo']`
        should also be defined.

        Furthermore, each subfilter must be a dictionary with one or both keys 'min' and
        'max'. These describe the exclusive minimum and maximum vales of the filtered
        data on that key.

        For example, if `cls.filters['obs']['foo'] = {'min': 0, 'max', 1000}`, and
        `adata` has been correctly annotated by `cls.annotate()`, then
        `bar=cls.filter(adata).obs['foo']` will be a filtered view of `adata` where the
        observations have been filtered to the range specified by `foo`, i.e.,
        `0<bar.obs['foo']<1000`.


        Returns
        -------
        Dict
            Complete filters used when applying `cls.filter(AnnData)`
        """
        tmp_filters = {"obs": {}, "var": {}}

        for base in cls.__mro__[::-1]:
            if hasattr((base_filters := getattr(base, "_filters", False)), "get"):
                for base_key in tmp_filters.keys():
                    tmp_filters[base_key].update(
                        {
                            k: v
                            for k, v in base_filters[base_key].items()
                            if v is not None or k not in tmp_filters[base_key]
                        }
                    )

        # check the filters for validity
        cls.__check_filters__(tmp_filters)

        return tmp_filters

    # Don't reimplement these if you can avoid it.

    def annotate(self, adata: AnnData) -> AnnData:
        """annotate: Apply annotations to AnnData object in place.

        Annotations are defined in subclasses and implementing classes by
        _annotate. This function traverses the implementing class MRO, building
        annotations from all bases that implement an _annotate method.

        Parameters
        ----------
        adata : AnnData
            Dataset to annotate

        Returns
        -------
        AnnData
            Annotated dataset
        """
        with self.logger.log_task("annotating AnnData"):
            for base in self.__class__.__mro__[::-1]:
                # annotate from the bottom of the mro up
                if (f := getattr(base, "_annotate", False)) and not hasattr(
                    f, "__isabstractmethod__"
                ):
                    adata = f(adata)
        return adata

    def filter(self, adata: AnnData) -> AnnData:
        """filter: Filter an annotated AnnData object according to `cls.filters`.

        Returns a filtered view of `adata`. To get an actual instance of the filtered
        data, use `cls.filter(adata).copy()`.

        For explanation of how filters are defined, see `Dataset.filters`.
        Parameters
        ----------
        adata : AnnData
            Annotated AnnData object.

        Returns
        -------
        AnnData
            Filtered view of `adata` according to cls.filters

        Raises
        ------
        KeyError
            `adata` is missing an annotation that is contained in `cls.filters`
        """

        mask = {
            "obs": np.ones(adata.shape[0]).astype(bool),
            "var": np.ones(adata.shape[1]).astype(bool),
        }
        filts = self.filters

        with self.logger.log_task("filtering AnnData"):
            for dimension_key, dimension_filters in filts.items():
                df = getattr(adata, dimension_key)
                for field, filt in dimension_filters.items():
                    if filt is None:
                        pass
                    else:
                        if field not in df.columns:
                            raise KeyError(
                                f"{field} is not a valid column in adata"
                                f".{dimension_key}."
                            )
                        # get the hi and low with defaults to inf.
                        hi = filt.get("max", np.Inf)
                        low = filt.get("min", -1 * np.Inf)
                        # update the mask
                        mask[dimension_key] &= df[field] >= low
                        mask[dimension_key] &= df[field] <= hi
            out = adata[mask["obs"], :][:, mask["var"]]
        return out

    def _get_files(
        self, paths: Dict[str, str], download: bool = True, overwrite: bool = False
    ):
        exceptions = {}
        to_download = {}
        to_copy = {}
        for local, remote in paths.items():
            local = str(local)
            remote = str(remote)
            if local == "." or remote == "." or local == "" or remote == "":
                exceptions[f"{local}: {remote}"] = IOError("Local path was '.'")
                continue
            else:
                local_path = Path(local).resolve()
                if overwrite or not local_path.exists():
                    if (remote_path := Path(remote).resolve()).exists():
                        to_copy[local_path] = remote_path
                    else:
                        if bool(urlparse(remote).scheme):
                            to_download[local_path] = remote
                    if local_path not in to_copy and local_path not in to_download:
                        exceptions[f"{local_path.name}: {remote}"] = IOError(
                            "Invalid path or URL."
                        )
        for local_path, remote_path in to_copy.items():
            if not local_path == remote_path:
                with self.logger.log_task(f"copying {remote_path} to {local_path}"):
                    try:
                        if not local_path.exists():
                            local_path.parents[0].mkdir(parents=True, exist_ok=True)
                        copyfile(remote_path, local_path)
                    except Exception as e:
                        exceptions[f"{str(local_path.name)}: {str(remote)}"] = e
            else:
                self.logger.log_info(
                    f"Skipping local path {local_path.name} because it"
                    " matches the remote."
                )

        if to_download and download:
            for path, url in to_download.items():
                with self.logger.log_task(f"downloading {url}"):
                    try:
                        download_url(url, path, logger=self.logger)
                    except Exception as e:
                        exceptions[f"{str(local_path.name)}: {str(remote)}"] = e
        if exceptions:
            handle_multiple_exceptions(IOError, "Unable to acquire files: ", exceptions)

    def acquire_raw_data(self, download: bool = True, overwrite: bool = False):
        with self.logger.log_task("acquiring raw data"):
            try:
                self._get_files(
                    {
                        v: self.__class__.raw_urls[k]
                        for k, v in self.raw_files_paths.items()
                    },
                    download=download,
                    overwrite=overwrite,
                )
            except Exception as e:
                self.logger.log_info(f"Unable to acquire raw files {e}")
                raise e

    def read_raw_data(self):
        with self.logger.log_task(
            "reading raw data from " f"{str(self.raw_data_path)}"
        ):
            try:
                adata = read_h5ad(str(self.raw_data_path))
            except FileNotFoundError as e:
                self.logger.log_info("Unable to retrieve raw data from disk.")
                raise e
        return adata

    def get_raw_data(
        self,
        download: bool = True,
        overwrite: bool = False,
        store_raw_data: Optional[bool] = None,
        store_raw_files: Optional[bool] = None,
    ) -> AnnData:
        """get_raw_data: Get the raw AnnData for the dataset.

        Attempts to retrieve from local storage based on `obj.raw_data_path`
        (defined at init).
        If unavailable, and `download == True`, then `get_raw_data` attempts to
        download the data from `cls.raw_data_url`, then process it into an
        adata.

        If `object.store_raw_data` then this function will write to
        `obj.raw_data_path`.
        If `object.store_raw_files`, then this function will not remove files
        generated during raw data processing.

        Parameters
        ----------
        download : bool, default True
            Retrieve data from internet if unavailable on local disk.
        store_raw_data : bool, optional
            Store raw data to disk. Defaults to `instance.store_raw_data`.
            Overwrites `instance.store_raw_data`.
        store_raw_files : bool, optional
            Store raw files to disk. Defaults to 'instance.store_raw_files`.
            Overwrites `instance.store_raw_files`.

        Returns
        -------
        AnnData
            Raw AnnData object containing the dataset.

        Raises
        ------
        RuntimeError
            Unable to get the raw data.
        """
        if store_raw_data is None:
            store_raw_data = self.store_raw_data
        else:
            self.store_raw_data = store_raw_data
        if store_raw_files is None:
            store_raw_files = self.store_raw_files
        else:
            self.store_raw_files = store_raw_files

        with self.logger.log_task("retrieving raw data"):

            try:
                adata = self.read_raw_data()
            except FileNotFoundError:
                try:
                    # we can still attempt to build the raw data if the data has been
                    # previously stored.
                    with self.logger.log_task("processing raw files"):
                        try:
                            # try to load the data
                            adata = self._process_raw_data()
                        except Exception:
                            # can't load the data
                            # try to download it
                            self.acquire_raw_data(
                                download=download, overwrite=overwrite
                            )
                            # try to load it again
                            adata = self._process_raw_data()

                    if store_raw_data:
                        with self.logger.log_task(f"writing {str(self.raw_data_path)}"):
                            adata.write(str(self.raw_data_path))
                    if not store_raw_files:
                        # clean the raw files out.
                        try:
                            self.logger.log_warning("Removing raw files.")
                            rmtree(self.raw_files_directory)
                        except Exception:
                            pass
                except Exception as exc2:
                    raise RuntimeError("`Unable to retrieve raw data. ") from exc2

            # run cleanup

        return adata

    def acquire_filtered_data(self, download: bool = True, overwrite: bool = False):
        with self.logger.log_task("acquiring filtered data"):
            try:
                self._get_files(
                    {self.filtered_data_path: self.__class__.filtered_url},
                    download=download,
                    overwrite=overwrite,
                )
            except Exception as e:
                self.logger.log_info(f"Unable to acquire filtered data {e}")
                raise e

    def read_filtered_data(self):
        with self.logger.log_task(
            "reading filtered data from " f"{str(self.filtered_data_path)}"
        ):
            try:
                adata = read_h5ad(str(self.filtered_data_path))

            except FileNotFoundError as e:
                self.logger.log_info("Unable to retrieve filtered data from disk.")
                raise e
        return adata

    def get_filtered_data(
        self,
        download: bool = True,
        overwrite: bool = False,
        store_filtered_data: Optional[bool] = None,
        **kwargs,
    ) -> AnnData:
        """get_filtered_data: Get the filtered adata for the dataset.

        Attempts to retrieve from local storage based on
         `instance.filtered_data_path` (defined at init).
        If unavailable, and `download == True`, then `get_filtered_data`
        attempts to download the data from `instance.filtered_data_url`.

        Finally, if the function still cannot retrieve the filtered data, it
        builds the data from the raw data by calling `obj.get_raw_data`.

        If `instance.store_filtered_data`, then this function will write to
        `instance.filtered_data_path`.
        Parameters

        Parameters
        ----------
        download : bool, default True
            Retrieve data from internet if unavailable on local disk.
        store_filtered_data : bool, optional
            Store filtered data to disk. Defaults to `instance.store_filtered_data`.
            Overwrites `instance.store_filtered_data`.

        Returns
        -------
        AnnData
            Filtered & annotated AnnData object containing the dataset.

        Raises
        ------
        RuntimeError
            Unable to get the filtered data and unable to get the raw data.
        """
        if store_filtered_data is None:
            store_filtered_data = self.store_filtered_data
        else:
            self.store_filtered_data = store_filtered_data
        adata = False
        with self.logger.log_task("retrieving filtered data"):
            try:  # read the filtered data from data_path
                adata = self.read_filtered_data()
            except FileNotFoundError:  # can't get from disk.
                try:
                    self.acquire_filtered_data(download=download, overwrite=overwrite)
                    adata = self.read_filtered_data()
                    if not store_filtered_data:
                        try:
                            self.logger.log_warning("Removing filtered file.")
                            rmtree(self.filtered_data_path)
                        except Exception:
                            pass
                except Exception:
                    pass
                if not adata:  # either download wasn't true or it failed.
                    # try to get the filtered data from raw
                    with self.logger.log_task("building filtered data from raw"):
                        adata = self.filter(
                            self.annotate(
                                self.get_raw_data(
                                    download=download, overwrite=overwrite, **kwargs
                                )
                            )
                        )
                    if store_filtered_data:
                        with self.logger.log_task(
                            f"writing {str(self.filtered_data_path)}"
                        ):
                            adata.write(self.filtered_data_path)

        return adata


class Modality(Dataset):
    """Modality: Abstract descendant of Dataset. Adds modality-level features to
    a final implementation of a Dataset.
    """

    def __init_subclass__(cls, **kwargs):
        """__init_subclass__: Initialize subclasses of `Modality`.

        This function sets `cls._modality=cls.__name__` when cls is a
        first-generation subclass of `Modality`.
        """
        super().__init_subclass__()

        if __class__ in cls.__bases__:
            cls._modality = cls.__name__

    @classproperty
    def modality(cls) -> str:
        """modality: The modality associated with a first generation subclass of
        Modality.

        `cls.modality` is set when a class is initialized that immediately bases
        `Modality`. For instance, if `cls` bases `base_cls`, and `base_cls` bases
        `Modality`, then `cls.modality = base_cls.modality`.
        Returns
        -------
        str
            The modality name

        Raises
        ------
        NotImplementedError
            If modality is not implemented for the calling class, e.g., the base
            class Modality.
        """
        if modality := getattr(cls, "_modality", False):
            return modality
        else:
            raise NotImplementedError(
                f"Property `modality` not implemented for {cls.__name__}"
            )


class Technology(ABC):
    """Technology: abstract mixin to add technology-wide features to a Dataset"""

    __abstractparents__ = {Modality}

    def __init_subclass__(cls, **kwargs):
        """__init_subclass__: Initialize subclasses of Technology.

        This function sets `cls._technology=cls.__name__` when cls is a
        first-generation subclass of `Technology`.
        """
        if __class__ in cls.__bases__:
            cls._technology = cls.__name__

    @classproperty
    def technology(cls):
        """technology: The modality associated with a 1st generation subclass of
        Technology.

        `cls.technology` is set when a class is initialized that immediately bases
        `Technology`. For instance, if `cls` bases `base_cls`, and `base_cls` bases
        `Technology`, then `cls.technology = base_cls.technology`.

        Returns
        -------
        str
            The technology name

        Raises
        ------
        NotImplementedError
            If technology is not implemented, e.g., the base
            class Technology.
        """
        if technology := getattr(cls, "_technology", False):
            return technology
        else:
            raise NotImplementedError(
                f"Property `technology` not implemented for {cls.__name__}"
            )

    @classproperty
    def technology_citation(cls):
        if citation := getattr(cls, "_technology_citation", False):
            return citation
        else:
            raise NotImplementedError(
                f"Technology citation not provided for {cls.__name__}"
            )
