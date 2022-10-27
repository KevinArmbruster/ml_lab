from sklearn.preprocessing import label_binarize
from util import *
from sklearn.model_selection import KFold


class RestrictedBoltzmannMachine():
    '''
    For more details : A Practical Guide to Training Restricted Boltzmann Machines https://www.cs.toronto.edu/~hinton/absps/guideTR.pdf
    '''

    def __init__(self, ndim_visible, ndim_hidden, is_bottom=False, image_size=[28, 28], is_top=False, n_labels=10,
                 batch_size=10):

        """
        Args:
          ndim_visible: Number of units in visible layer.
          ndim_hidden: Number of units in hidden layer.
          is_bottom: True only if this rbm is at the bottom of the stack in a deep belief net. Used to interpret visible layer as image data with dimensions "image_size".
          image_size: Image dimension for visible layer.
          is_top: True only if this rbm is at the top of stack in deep beleif net. Used to interpret visible layer as concatenated with "n_label" unit of label data at the end. 
          n_label: Number of label categories.
          batch_size: Size of mini-batch.
        """

        self.ndim_visible = ndim_visible

        self.ndim_hidden = ndim_hidden

        self.is_bottom = is_bottom

        if is_bottom: self.image_size = image_size

        self.is_top = is_top

        if is_top: self.n_labels = 10

        self.batch_size = batch_size

        self.delta_bias_v = 0

        self.delta_weight_vh = 0

        self.delta_bias_h = 0

        self.bias_v = np.random.normal(loc=0.0, scale=0.01, size=(self.ndim_visible))

        self.weight_vh = np.random.normal(loc=0.0, scale=0.01, size=(self.ndim_visible, self.ndim_hidden))

        self.bias_h = np.random.normal(loc=0.0, scale=0.01, size=(self.ndim_hidden))

        self.delta_weight_v_to_h = 0

        self.delta_weight_h_to_v = 0

        self.weight_v_to_h = None

        self.weight_h_to_v = None

        self.learning_rate = 0.01

        self.momentum = 0.7

        self.weight_cost = 0.0001

        self.print_period = 5000

        self.rf = {  # receptive-fields. Only applicable when visible layer is input data
            "period": 5000,  # iteration period to visualize
            "grid": [5, 5],  # size of the grid
            "ids": np.random.randint(0, self.ndim_hidden, 25)  # pick some random hidden units
        }

        self.weight_histogram_period = 5000

        return

    def cd1(self, visible_trainset, n_iterations=10000):

        """Contrastive Divergence with k=1 full alternating Gibbs sampling

        Args:
          visible_trainset: training data for this rbm, shape is (size of training set, size of visible layer)
          n_iterations: number of iterations of learning (each iteration learns a mini-batch)
        """

        print("learning CD1")

        n_samples = visible_trainset.shape[0]

        for it in range(n_iterations + 1):  # +1 for printing with %-operations

            # [TODO TASK 4.1] run k=1 alternating Gibbs sampling : v_0 -> h_0 -> v_1 -> h_1.
            # you may need to use the inference functions 'get_h_given_v' and 'get_v_given_h'.
            # note that inference methods returns both probabilities and activations (samples from probablities)
            # and you may have to decide when to use what.

            ### Random sampling of minibatch
            random_rows = np.random.randint(n_samples, size=self.batch_size)
            minibatch = visible_trainset[random_rows, :]

            ### Gibbs sampling
            p_v_h_0 = minibatch
            p_h_v_0, h_0 = self.get_h_given_v(p_v_h_0)
            p_v_h_1, v_1 = self.get_v_given_h(h_0)
            p_h_v_1, h_1 = self.get_h_given_v(p_v_h_1)

            # [TODO TASK 4.1] update the parameters using function 'update_params'
            self.update_params(p_v_h_0, h_0, p_v_h_1, p_h_v_1)

            ### Visualizations
            # visualize once in a while when visible layer is input images
            if it % self.rf["period"] == 0 and self.is_bottom:
                viz_rf(weights=self.weight_vh[:, self.rf["ids"]].reshape((self.image_size[0], self.image_size[1], -1)),
                       it=it, grid=self.rf["grid"])

            # visualize weights as histogram
            if it % self.rf["period"] == 0:
                weight_histogram(weights_vh=self.weight_vh, bias_h=self.bias_h, bias_v=self.bias_v, it=it)

            # print progress
            if it % self.print_period == 0:
                print("iteration=%7d recon_loss=%4.4f" % (it, self.calc_reconstruction_error(visible_trainset)))

        return

    def cd1_p(self, visible_trainset, n_iterations=10000):
        
        """Contrastive Divergence with k=1 full alternating Gibbs sampling

        Args:
          visible_trainset: training data for this rbm, shape is (size of training set, size of visible layer)
          n_iterations: number of iterations of learning (each iteration learns a mini-batch)
        """

        print("learning CD1")

        n_samples = visible_trainset.shape[0]

        for it in range(n_iterations + 1):  # +1 for printing with %-operations

            # [TODO TASK 4.1] run k=1 alternating Gibbs sampling : v_0 -> h_0 -> v_1 -> h_1.
            # you may need to use the inference functions 'get_h_given_v' and 'get_v_given_h'.
            # note that inference methods returns both probabilities and activations (samples from probablities)
            # and you may have to decide when to use what.

            ### Random sampling of minibatch
            random_rows = np.random.randint(n_samples, size=self.batch_size)
            minibatch = visible_trainset[random_rows, :]

            ### Gibbs sampling
            p_v_h_0 = minibatch
            p_h_v_0, h_0 = self.get_h_given_v(p_v_h_0)
            p_v_h_1, v_1 = self.get_v_given_h(h_0)
            p_h_v_1, h_1 = self.get_h_given_v(p_v_h_1)

            # [TODO TASK 4.1] update the parameters using function 'update_params'
            self.update_params(p_v_h_0, h_0, p_v_h_1, p_h_v_1)

            ### Visualizations
            # visualize once in a while when visible layer is input images
            if it % self.rf["period"] == 0 and self.is_bottom:
                viz_rf_p(weights=self.weight_vh[:, self.rf["ids"]].reshape((self.image_size[0], self.image_size[1], -1)),
                       it=it, grid=self.rf["grid"])

            # visualize weights as histogram
            if it % self.rf["period"] == 0:
                weight_histogram(weights_vh=self.weight_vh, bias_h=self.bias_h, bias_v=self.bias_v, it=it)

            # print progress
            if it % self.print_period == 0:
                print("iteration=%7d recon_loss=%4.4f" % (it, self.calc_reconstruction_error(visible_trainset)))

        return

    def calc_reconstruction(self, data):
        n_samples = data.shape[0]

        ### basically same as cd1, just shortened
        p_h_v_0, h_0 = self.get_h_given_v(data)
        p_v_h_1, v_1 = self.get_v_given_h(h_0)
        return p_v_h_1

    def calc_reconstruction_error(self, data):
        n_samples = data.shape[0]

        recon = self.calc_reconstruction(data)

        error = np.linalg.norm(data - recon)
        return error / n_samples

    def update_params(self, v_0, h_0, v_k, h_k):

        """Update the weight and bias parameters.

        You could also add weight decay and momentum for weight updates.

        Args:
           v_0: activities or probabilities of visible layer (data to the rbm)
           h_0: activities or probabilities of hidden layer
           v_k: activities or probabilities of visible layer
           h_k: activities or probabilities of hidden layer
           all args have shape (size of mini-batch, size of respective layer)
        """

        # [TODO TASK 4.1] get the gradients from the arguments (replace the 0s below) and update the weight and bias parameters
        n_samples = v_0.shape[0]
        n_dim_visible = v_0.shape[1]
        n_dim_hidden = h_0.shape[1]

        ### regularization - use only 1 form!
        # (visible, hidden)
        l2_weight_decay = 2 * self.weight_cost * self.weight_vh
        # l1_lasso        = 2 * self.weight_cost * np.sign(self.weight_vh)

        ### calculate gradient
        # (visible, hidden) = (visible, sample) @ (sample, hidden) - same
        gradient_tmp = v_0.T @ h_0 - v_k.T @ h_k
        weight_vh_gradient = (1 / n_samples * gradient_tmp) + l2_weight_decay

        ### calculate weight update with momentum
        self.delta_bias_v    = self.momentum * self.delta_bias_v    + (1 - self.momentum) * np.sum(v_0 - v_k, axis=0)
        self.delta_weight_vh = self.momentum * self.delta_weight_vh + (1 - self.momentum) * weight_vh_gradient
        self.delta_bias_h    = self.momentum * self.delta_bias_h    + (1 - self.momentum) * np.sum(h_0 - h_k, axis=0)

        assert self.delta_bias_v.shape == (n_dim_visible,)
        assert self.delta_weight_vh.shape == (n_dim_visible, n_dim_hidden)
        assert self.delta_bias_h.shape == (n_dim_hidden,)

        ### update weights, use learning rate and normalize by sample size
        self.bias_v += self.learning_rate * self.delta_bias_v / n_samples
        self.weight_vh += self.learning_rate * self.delta_weight_vh / n_samples
        self.bias_h += self.learning_rate * self.delta_bias_h / n_samples

        return

    def get_h_given_v(self, visible_minibatch):

        """Compute probabilities p(h|v) and activations h ~ p(h|v) 

        Uses undirected weight "weight_vh" and bias "bias_h"
        
        Args: 
           visible_minibatch: shape is (size of mini-batch, size of visible layer)
        Returns:        
           tuple ( p(h|v) , h) 
           both are shaped (size of mini-batch, size of hidden layer)
        """

        assert self.weight_vh is not None

        n_samples = visible_minibatch.shape[0]

        # [TODO TASK 4.1] compute probabilities and activations (samples from probabilities) of hidden layer (replace the zeros below)
        # (1, self.ndim_hidden) - (n_samples, ndim_visible) @ (ndim_visible, ndim_hidden)
        support = self.bias_h + visible_minibatch @ self.weight_vh
        p_h_v = sigmoid(support)

        h = sample_binary(p_h_v)

        assert p_h_v.shape == (n_samples, self.ndim_hidden)
        assert h.shape == (n_samples, self.ndim_hidden)
        return p_h_v, h

    def get_v_given_h(self, hidden_minibatch):

        """Compute probabilities p(v|h) and activations v ~ p(v|h)

        Uses undirected weight "weight_vh" and bias "bias_v"
        
        Args: 
           hidden_minibatch: shape is (size of mini-batch, size of hidden layer)
        Returns:        
           tuple ( p(v|h) , v) 
           both are shaped (size of mini-batch, size of visible layer)
        """

        assert self.weight_vh is not None

        n_samples = hidden_minibatch.shape[0]

        if self.is_top:

            """
            Here visible layer has both data and labels. Compute total input for each unit (identical for both cases), \ 
            and split into two parts, something like support[:, :-self.n_labels] and support[:, -self.n_labels:]. \
            Then, for both parts, use the appropriate activation function to get probabilities and a sampling method \
            to get activities. The probabilities as well as activities can then be concatenated back into a normal visible layer.
            """

            # [TODO TASK 4.1] compute probabilities and activations (samples from probabilities) of visible layer (replace the pass below). \
            # Note that this section can also be postponed until TASK 4.2, since in this task, stand-alone RBMs do not contain labels in visible layer.

            support = self.bias_v + hidden_minibatch @ self.weight_vh.T       
            
            p_v_h = sigmoid(support)  
            labels = p_v_h[:, -self.n_labels:]  
            labels = softmax(labels)

            v = sample_binary(p_v_h[:, :-self.n_labels])
            labels = sample_categorical(labels)

            v = np.concatenate((v, labels), axis = 1)

        else:

            # [TODO TASK 4.1] compute probabilities and activations (samples from probabilities) of visible layer (replace the pass and zeros below)
            # (1, self.ndim_visible) - (ndim_visible, ndim_hidden) @ (ndim_hidden, n_samples)
            support = self.bias_v + hidden_minibatch @ self.weight_vh.T
            p_v_h = sigmoid(support)

            v = sample_binary(p_v_h)

        assert p_v_h.shape == (n_samples, self.ndim_visible)
        assert v.shape == (n_samples, self.ndim_visible)
        return p_v_h, v

    """ rbm as a belief layer : the functions below do not have to be changed until running a deep belief net """

    def untwine_weights(self):

        self.weight_v_to_h = np.copy(self.weight_vh)
        self.weight_h_to_v = np.copy(np.transpose(self.weight_vh))

    def get_h_given_v_dir(self, visible_minibatch):

        """Compute probabilities p(h|v) and activations h ~ p(h|v)

        Uses directed weight "weight_v_to_h" and bias "bias_h"
        
        Args: 
           visible_minibatch: shape is (size of mini-batch, size of visible layer)
        Returns:        
           tuple ( p(h|v) , h) 
           both are shaped (size of mini-batch, size of hidden layer)
        """

        assert self.weight_v_to_h is not None

        n_samples = visible_minibatch.shape[0]

        # [TODO TASK 4.2] perform same computation as the function 'get_h_given_v' but with directed connections (replace the zeros below) 
        support = self.bias_h + visible_minibatch @ self.weight_v_to_h
        p_h_v = sigmoid(support)

        h = sample_binary(p_h_v)

        assert p_h_v.shape == (n_samples, self.ndim_hidden)
        assert h.shape == (n_samples, self.ndim_hidden)
        return p_h_v, h
        #return np.zeros((n_samples, self.ndim_hidden)), np.zeros((n_samples, self.ndim_hidden))

    def get_v_given_h_dir(self, hidden_minibatch):

        """Compute probabilities p(v|h) and activations v ~ p(v|h)

        Uses directed weight "weight_h_to_v" and bias "bias_v"
        
        Args: 
           hidden_minibatch: shape is (size of mini-batch, size of hidden layer)
        Returns:        
           tuple ( p(v|h) , v) 
           both are shaped (size of mini-batch, size of visible layer)
        """

        assert self.weight_h_to_v is not None

        n_samples = hidden_minibatch.shape[0]

        if self.is_top:

            """
            Here visible layer has both data and labels. Compute total input for each unit (identical for both cases), \ 
            and split into two parts, something like support[:, :-self.n_labels] and support[:, -self.n_labels:]. \
            Then, for both parts, use the appropriate activation function to get probabilities and a sampling method \
            to get activities. The probabilities as well as activities can then be concatenated back into a normal visible layer.
            """

            # [TODO TASK 4.2] Note that even though this function performs same computation as 'get_v_given_h' but with directed connections,
            # this case should never be executed : when the RBM is a part of a DBN and is at the top, it will have not have directed connections.
            # Appropriate code here is to raise an error (replace pass below)

            raise Exception("It seems to be an error here") 

        else:
    
            # [TODO TASK 4.1] compute probabilities and activations (samples from probabilities) of visible layer (replace the pass and zeros below)
            # (1, self.ndim_visible) - (ndim_visible, ndim_hidden) @ (ndim_hidden, n_samples)
            support = self.bias_v + hidden_minibatch @ self.weight_h_to_v.T
            p_v_h = sigmoid(support)

            v = sample_binary(p_v_h)

        assert p_v_h.shape == (n_samples, self.ndim_visible)
        assert v.shape == (n_samples, self.ndim_visible)
        return p_v_h, v

        #return np.zeros((n_samples, self.ndim_visible)), np.zeros((n_samples, self.ndim_visible))

    def update_generate_params(self, inps, trgs, preds):

        """Update generative weight "weight_h_to_v" and bias "bias_v"
        
        Args:
           inps: activities or probabilities of input unit
           trgs: activities or probabilities of output unit (target)
           preds: activities or probabilities of output unit (prediction)
           all args have shape (size of mini-batch, size of respective layer)
        """

        # [TODO TASK 4.3] find the gradients from the arguments (replace the 0s below) and update the weight and bias parameters.

        self.delta_weight_h_to_v += 0
        self.delta_bias_v += 0

        self.weight_h_to_v += self.delta_weight_h_to_v
        self.bias_v += self.delta_bias_v

        return

    def update_recognize_params(self, inps, trgs, preds):

        """Update recognition weight "weight_v_to_h" and bias "bias_h"
        
        Args:
           inps: activities or probabilities of input unit
           trgs: activities or probabilities of output unit (target)
           preds: activities or probabilities of output unit (prediction)
           all args have shape (size of mini-batch, size of respective layer)
        """

        # [TODO TASK 4.3] find the gradients from the arguments (replace the 0s below) and update the weight and bias parameters.

        self.delta_weight_v_to_h += 0
        self.delta_bias_h += 0

        self.weight_v_to_h += self.delta_weight_v_to_h
        self.bias_h += self.delta_bias_h

        return
