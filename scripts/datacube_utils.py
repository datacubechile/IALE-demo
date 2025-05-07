#!python3

# A collection of utilities that can be used in with the Open Data Cube API.
#
# License: Apache 2.0

# Created for EASI Hub training notebooks, https://dev.azure.com/csiro-easi/easi-hub-public/_git/hub-notebooks


# Borrowed from https://github.com/GeoscienceAustralia/dea-notebooks/blob/develop/Tools/dea_tools/datahandling.py
def dc_query_only(**kw):
    """
    Remove load-only parameters, the rest can be passed to Query
    Returns
    -------
    dict of query parameters
    """

    def _impl(measurements=None,
              output_crs=None,
              resolution=None,
              resampling=None,
              skip_broken_datasets=None,
              dask_chunks=None,
              fuse_func=None,
              align=None,
              datasets=None,
              progress_cbk=None,
              group_by=None,
              **query):
        return query

    return _impl(**kw)