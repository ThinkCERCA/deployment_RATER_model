def filter_claim_and_evidence(data):
    """
    Filters and returns only the 'claim' and 'evidence' fields from the input dictionary.

    Args:
        data (dict): The input JSON dictionary with keys 'claim', 'point', and 'evidence'.

    Returns:
        dict: A dictionary containing only 'claim' and 'evidence' fields.
    """
    # Extract only 'claim' and 'evidence' fields
    return {key: data[key] for key in ['Claim', 'Evidence','Counterclaim','Rebuttal','Concluding Statement'] if key in data}