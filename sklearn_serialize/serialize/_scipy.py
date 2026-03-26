from scipy import sparse

from ._core import RESTORE_FUNCTION_FACTORY, restore, serialize

# Register most-specific subclasses before the spmatrix base to ensure
# singledispatch picks the right handler via MRO.


@serialize.register(sparse.coo_matrix)
def serialize_sparse_coo_matrix(data):
    return {
        "py/scipy.sparse": {
            "format": data.getformat(),
            "data": serialize(data.data),
            "row": serialize(data.row),
            "col": serialize(data.col),
            "shape": data.shape,
        }
    }


@serialize.register(sparse.lil_matrix)
@serialize.register(sparse.dok_matrix)
def serialize_sparse_dense_matrix(data):
    # lil and dok have no .indices/.indptr; serialize via dense roundtrip
    return {
        "py/scipy.sparse": {
            "format": data.getformat(),
            "dense": serialize(data.toarray()),
            "shape": data.shape,
        }
    }


@serialize.register(sparse.spmatrix)
def serialize_sparse_matrix(data):
    return {
        "py/scipy.sparse": {
            "data": serialize(data.data),
            "indices": serialize(data.indices),
            "indptr": serialize(data.indptr),
            "shape": data.shape,
            "format": data.getformat(),
        }
    }


def restore_sparse_matrix(dct):
    from scipy.sparse import coo_matrix, csc_matrix, csr_matrix, dok_matrix, lil_matrix

    sparse_format_factory = {
        "csr": csr_matrix,
        "csc": csc_matrix,
        "coo": coo_matrix,
        "lil": lil_matrix,
        "dok": dok_matrix,
    }
    data = dct["py/scipy.sparse"]
    format = data["format"]
    shape = tuple(data["shape"])
    constructor = sparse_format_factory.get(format)

    if format in ["csr", "csc"]:
        matrix = constructor(
            (restore(data["data"]), restore(data["indices"]), restore(data["indptr"])),
            shape=shape,
        )
    elif format == "coo":
        matrix = constructor(
            (restore(data["data"]), (restore(data["row"]), restore(data["col"]))),
            shape=shape,
        )
    else:
        matrix = constructor(restore(data["dense"]), shape=shape)

    return matrix


RESTORE_FUNCTION_FACTORY["py/scipy.sparse"] = restore_sparse_matrix
