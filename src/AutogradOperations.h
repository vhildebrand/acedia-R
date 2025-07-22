#ifndef AUTOGRAD_OPERATIONS_H
#define AUTOGRAD_OPERATIONS_H

#include "ComputationGraph.h"
#include "TensorRegistry.h"
#include <memory>

/**
 * @brief Autograd-aware tensor operations that build computation graphs
 */
class AutogradOps {
public:
    // Addition with autograd
    static std::shared_ptr<GraphNode> add(std::shared_ptr<GraphNode> a, std::shared_ptr<GraphNode> b) {
        auto result_node = std::make_shared<AddNode>(a, b);
        
        // Perform forward pass
        auto result_tensor = a->get_tensor()->add(*b->get_tensor());
        result_node->set_output_tensor(result_tensor);
        
        return result_node;
    }
    
    // Scalar multiplication with autograd
    static std::shared_ptr<GraphNode> scalar_mul(std::shared_ptr<GraphNode> input, double scalar) {
        auto result_node = std::make_shared<ScalarMulNode>(input, scalar);
        
        // Perform forward pass
        auto result_tensor = input->get_tensor()->scalar_mul(scalar);
        result_node->set_output_tensor(result_tensor);
        
        return result_node;
    }
    
    // Element-wise multiplication with autograd
    static std::shared_ptr<GraphNode> mul(std::shared_ptr<GraphNode> a, std::shared_ptr<GraphNode> b) {
        auto result_node = std::make_shared<MulNode>(a, b);
        
        // Perform forward pass
        auto result_tensor = a->get_tensor()->mul(*b->get_tensor());
        result_node->set_output_tensor(result_tensor);
        
        return result_node;
    }
    
    // Matrix multiplication with autograd
    static std::shared_ptr<GraphNode> matmul(std::shared_ptr<GraphNode> a, std::shared_ptr<GraphNode> b) {
        auto result_node = std::make_shared<MatMulNode>(a, b);
        
        // Perform forward pass
        auto result_tensor = a->get_tensor()->matmul(*b->get_tensor());
        result_node->set_output_tensor(result_tensor);
        
        return result_node;
    }
    
    // Create a leaf node from a tensor
    static std::shared_ptr<GraphNode> create_leaf(std::shared_ptr<TensorBase> tensor, bool requires_grad = false) {
        return std::make_shared<LeafNode>(tensor, requires_grad);
    }
    
    // Execute backward pass from a node
    static void backward(std::shared_ptr<GraphNode> node) {
        ComputationGraph graph;
        graph.backward(node);
    }
};

/**
 * @brief Convenience macros for no_grad contexts
 */
#define NO_GRAD() GradContext::NoGradGuard no_grad_guard

/**
 * @brief Higher-level operations for common ML patterns
 */
class MLOperations {
public:
    // Sigmoid activation with autograd
    static std::shared_ptr<GraphNode> sigmoid(std::shared_ptr<GraphNode> input);
    
    // ReLU activation with autograd  
    static std::shared_ptr<GraphNode> relu(std::shared_ptr<GraphNode> input);
    
    // Cross-entropy loss with autograd
    static std::shared_ptr<GraphNode> cross_entropy_loss(
        std::shared_ptr<GraphNode> predictions,
        std::shared_ptr<GraphNode> targets
    );
    
    // Mean squared error loss with autograd
    static std::shared_ptr<GraphNode> mse_loss(
        std::shared_ptr<GraphNode> predictions, 
        std::shared_ptr<GraphNode> targets
    );
    
    // Sum reduction with autograd
    static std::shared_ptr<GraphNode> sum(std::shared_ptr<GraphNode> input);
    
    // Mean reduction with autograd
    static std::shared_ptr<GraphNode> mean(std::shared_ptr<GraphNode> input);
};

#endif // AUTOGRAD_OPERATIONS_H 