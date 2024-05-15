import numpy as np
import panphon
from ipapy import is_valid_ipa

ipa_feature_table = panphon.FeatureTable()


def phonetic_distance(pa: str, pb: str, relative: bool = False) -> int | float:
    """
    Computes the phonetic (hamming) distance between two phonemes
    based on their articulary features as defined in panphon.
    """
    feat_a = np.array(ipa_feature_table.word_to_vector_list(pa, numeric=True)).ravel()
    feat_b = np.array(ipa_feature_table.word_to_vector_list(pb, numeric=True)).ravel()
    dist = (feat_a != feat_b).sum().astype("int").sum()
    if relative:
        dist = dist / feat_a.shape[0]
    return dist


def get_inventory_mapping(
    src_inventory: set[str], trgt_inventory: set[str], include_distances: bool = False
) -> dict[str]:
    src_inventory, trgt_inventory = map(list, (src_inventory, trgt_inventory))

    def find_closest_phoneme(src_p):
        if src_p in trgt_inventory:
            if include_distances:
                return (src_p, 0)
            return src_p
        else:
            distances = [
                phonetic_distance(src_p, trgt_p)
                for trgt_p in trgt_inventory
                if is_valid_ipa(trgt_p)
            ]
            closest_idx = np.argmin(distances)
            if include_distances:
                return trgt_inventory[closest_idx], distances[closest_idx]
            return trgt_inventory[closest_idx]

    inventory_map = {
        src_p: find_closest_phoneme(src_p)
        for src_p in src_inventory
        if is_valid_ipa(src_p)
    }
    return inventory_map
