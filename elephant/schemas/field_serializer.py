import quantities as pq

def serialize_quantity(value: pq.Quantity) -> dict:
    if value is None:
        return None
    return {
        "value": value.magnitude,
        "unit": value.dimensionality
    }