import torch
import random

class FPLServer:
    def __init__(self, subset_size=3):
        self.global_prototypes = {}
        self.subset_size = subset_size # Optimal config from paper is 3

    def aggregate_prototypes(self, client_prototypes_list):
        """
        Receives a list of local prototype dictionaries from all clients.
        Instead of aggregating all of them, it randomly selects `subset_size`
        prototypes per class to compute the global prototype without causing stray issues.
        """
        # Re-organize locally reported prototypes by their Class
        class_buckets = {}
        for c_protos in client_prototypes_list:
            for class_idx, proto_vec in c_protos.items():
                if class_idx not in class_buckets:
                    class_buckets[class_idx] = []
                class_buckets[class_idx].append(proto_vec)

        # Aggregate for each class bucket randomly
        new_global_prototypes = {}
        for class_idx, protos in class_buckets.items():
            if len(protos) > 0:
                # Randomly pick subset
                selection_size = min(self.subset_size, len(protos))
                selected_protos = random.sample(protos, selection_size)
                
                # Average them
                stacked = torch.stack(selected_protos)
                new_global_prototypes[class_idx] = torch.mean(stacked, dim=0)

        self.global_prototypes = new_global_prototypes
        return new_global_prototypes
