import torch
import torch.nn as nn
import torch.optim as optim

class FPLClient:
    def __init__(self, client_id, model, X_train, y_train, X_test, y_test, lr=0.01, lam=0.1):
        self.client_id = client_id
        self.model = model
        
        self.X_train = torch.tensor(X_train, dtype=torch.float32)
        self.y_train = torch.tensor(y_train, dtype=torch.long)
        
        self.X_test = torch.tensor(X_test, dtype=torch.float32)
        self.y_test = torch.tensor(y_test, dtype=torch.long)
        
        self.lr = lr
        self.lam = lam  # Regularization weighting factor
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        
        # Compute class weights for severe class imbalance
        num_neg = (self.y_train == 0).sum().item()
        num_pos = (self.y_train == 1).sum().item()
        weight_pos = float(num_neg) / float(max(1, num_pos))
        class_weights = torch.tensor([1.0, weight_pos], dtype=torch.float32)
        
        self.criterion_S = nn.CrossEntropyLoss(weight=class_weights) # Supervised Loss LMS
        self.criterion_R = nn.MSELoss()          # Prototype Regularization L_R
        
        # Local Prototypes mapping class index -> prototype vector
        self.local_prototypes = {}

    def get_local_prototypes(self):
        """ Calculate local prototypes by averaging the embedding vector per class. """
        self.model.eval()
        prototypes = {}
        with torch.no_grad():
            _, embeddings = self.model(self.X_train)
            
            for class_idx in torch.unique(self.y_train):
                class_mask = (self.y_train == class_idx)
                class_embeddings = embeddings[class_mask]
                if len(class_embeddings) > 0:
                    prototype = torch.mean(class_embeddings, dim=0)
                    prototypes[class_idx.item()] = prototype

        self.local_prototypes = prototypes
        return prototypes

    def local_train(self, global_prototypes, epochs=5):
        """ Run E local epochs balancing supervised loss and global prototype regularization """
        self.model.train()
        
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            
            logits, embeddings = self.model(self.X_train)
            
            # 1. Supervised Learning loss
            loss_S = self.criterion_S(logits, self.y_train)
            
            # 2. Prototype Regularization Loss
            loss_R = 0.0
            if global_prototypes is not None:
                # We iteratively calculate the mse distance of embeddings against their corresponding global prototype
                for c, g_proto in global_prototypes.items():
                    class_mask = (self.y_train == c)
                    class_embeddings = embeddings[class_mask]
                    if len(class_embeddings) > 0:
                        # Replicate global prototype to match shape of class_embeddings
                        g_proto_tensor = g_proto.repeat(class_embeddings.size(0), 1)
                        loss_R += self.criterion_R(class_embeddings, g_proto_tensor)
            
            # Total Loss
            total_loss = loss_S + self.lam * loss_R
            total_loss.backward()
            self.optimizer.step()
            
        # Re-calc local prototype upon completing local update chunk
        return self.get_local_prototypes()
        
    def evaluate(self, global_prototypes):
        """ Phase 4 Inference: Shortest Euclidean Distance to Prototype """
        if not global_prototypes:
            return 0.0, 0.0 # Cannot test if server has no prototypes
            
        self.model.eval()
        with torch.no_grad():
            _, embeddings = self.model(self.X_test)
            
            predictions = []
            probabilities = []
            for emb in embeddings:
                dist_0 = float('inf')
                dist_1 = float('inf')
                
                if 0 in global_prototypes:
                    dist_0 = torch.norm(emb - global_prototypes[0], p=2).item()
                if 1 in global_prototypes:
                    dist_1 = torch.norm(emb - global_prototypes[1], p=2).item()
                    
                pred_label = 0 if dist_0 < dist_1 else 1
                predictions.append(pred_label)
                
                # Convert to probabilistic score for AUC
                total_dist = dist_0 + dist_1
                if total_dist == 0:
                    prob_1 = 0.5
                else:
                    prob_1 = dist_0 / total_dist
                probabilities.append(prob_1)
                
            predictions = torch.tensor(predictions)
            probabilities = torch.tensor(probabilities)
            
            # Simple Accuracy metric 
            correct = (predictions == self.y_test).sum().item()
            acc = correct / len(self.y_test)
            
        return acc, predictions, probabilities
