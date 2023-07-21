class SoftKMeans(torch.nn.Module):
    
    def __init__(self, k: int, beta: float = 10.0, p: float = 2, center: bool = True, scale: bool = True, max_iters: int = 100, threshold: float = 1e-3):
        super(SoftKMeans, self).__init__()
        '''
        A clusterer object that can be called like any other pytorch module to produce soft cluster assignments using soft K-means.
            
            Parameters:
                k (int) : Number of clusters to be considered
                beta (float) : The "stiffness" parameter. Determines how decisively points are assigned to clusters
                p (float) : The p used for the Lp-norm distance function
                center (bool) : Flag for if data dimensions are centered to 0 in clustering
                scale (bool) : Flag for if data dimensions are rescaled to have a standard deviation of 1. Helps with dimension importance
                max_iters (int) : Maximum number of iterations for the algorithm
                threshold (float) : Threshold value for early stopping
        '''
        self.k = k
        self.beta = beta
        self.p = p
        self.center = center
        self.scale = scale
        self.max_iters = max_iters
        self.threshold = threshold
    
    def forward(self, points: torch.Tensor, return_centers: bool = False):
        '''
        Given a tensor of points, return their soft assignments and optional cluster centers
            
            Parameters:
                points (torch.Tensor) : The points to be clustered. Expected shape is [N, d]
                return_centers (bool) : Flag for if cluster centers are returned or not
            
            Returns:
                assignments (torch.Tensor) : Probabilities for cluster assignments of shape [N, k]
                centers (torch.Tensor) : Optional cluster centers found of shape [k, d]
        '''
        
        # Step 0: normalise data as needed
        std, mean = torch.std_mean(points, dim=0)
        if self.center:
            points = points - mean
        if self.scale:
            points = points / (std + 1e-5) # Oh I'm so stable
        
        # Step 1: generate the initial cluster centers
        centers = points.mean(dim=0, keepdim=True).expand(self.k, -1) # shape [k, d]
        centers = centers + torch.normal(torch.zeros_like(centers), 0.5)
        
        # Step 2: do the actual clustering
        prev_centers = centers.detach().clone()
        assigments = None
        
        for i in range(self.max_iters):
            # First, an "E" step - get new assigments
            # Get distances from each point to each center
            diff_vecs = points.unsqueeze(1).expand(-1, self.k, -1) - centers.unsqueeze(0) # shape [N, k, d]
            dists = torch.linalg.vector_norm(diff_vecs, ord=self.p, dim=-1) # shape [N, k]
            # Then weight and softmax
            assignments = F.softmax(-1 * self.beta * dists, dim=1)
            
            # Next, the "M" step - get new centers
            # Centers are weighted means of the new assignments
            centers = (assignments.T @ points) / assignments.sum(dim=0).unsqueeze(dim=1)
            
            if (centers-prev_centers).pow(2).sum().pow(0.5) < self.threshold:
                print(i)
                break
            else:
                prev_centers = centers.detach().clone()

        if return_centers:
            if self.scale:
                centers = centers * std
            if self.center:
                centers = centers + mean
            return assignments, centers
        return assignments
