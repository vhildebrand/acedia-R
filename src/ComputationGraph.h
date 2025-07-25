#ifndef COMPUTATION_GRAPH_H
#define COMPUTATION_GRAPH_H

#include <memory>
#include <vector>
#include <string>
#include <functional>
#include <unordered_set>
#include <stdexcept>
#include "TensorRegistry.h"  // Added for tensor utilities

// Forward declarations
template<typename T> class gpuTensor;
class TensorBase;

/**
 * @brief Abstract base class for all operations in the computation graph
 */
class GraphNode {
public:
    virtual ~GraphNode() = default;
    
    // Execute backward pass for this operation
    virtual void backward() = 0;ww
    
    // Get the operation name (for debugging/visualization)
    virtual std::string name() const = 0;
    
    // Get input nodes
    virtual std::vector<std::shared_ptr<GraphNode>> inputs() const = 0;
    
    // Set gradient for leaf nodes
    virtual void set_gradient(std::shared_ptr<TensorBase> grad) = 0;
    virtual std::shared_ptr<TensorBase> get_gradient() const = 0;
    
    // Check if this node requires gradients
    virtual bool requires_grad() const = 0;
    
    // Get the tensor associated with this node
    virtual std::shared_ptr<TensorBase> get_tensor() const = 0;

    // New: clear stored gradient (optional override)
    virtual void clear_gradient() { /* default no-op */ }
};

/**
 * @brief Leaf node representing a tensor that doesn't depend on other operations
 */
class LeafNode : public GraphNode {
private:
    std::shared_ptr<TensorBase> tensor_;
    std::shared_ptr<TensorBase> gradient_;
    bool requires_grad_;

public:
    LeafNode(std::shared_ptr<TensorBase> tensor, bool requires_grad = false)
        : tensor_(tensor), requires_grad_(requires_grad) {}
    
    void backward() override {
        // Leaf nodes don't propagate gradients further
    }
    
    std::string name() const override { return "LeafNode"; }
    
    std::vector<std::shared_ptr<GraphNode>> inputs() const override {
        return {}; // Leaf nodes have no inputs
    }
    
    void set_gradient(std::shared_ptr<TensorBase> grad) override {
        if (gradient_) {
            // Accumulate gradients if already exists
            auto accumulated = gradient_->add(*grad);
            gradient_.reset(accumulated.release());
        } else {
            gradient_ = grad;
        }
    }
    
    std::shared_ptr<TensorBase> get_gradient() const override {
        return gradient_;
    }
    
    bool requires_grad() const override { return requires_grad_; }
    
    std::shared_ptr<TensorBase> get_tensor() const override { return tensor_; }

    void clear_gradient() override {
        gradient_.reset();
    }
};

/**
 * @brief Function node representing an operation with forward and backward passes
 */
class FunctionNode : public GraphNode {
protected:
    std::vector<std::shared_ptr<GraphNode>> inputs_;
    std::shared_ptr<TensorBase> output_tensor_;
    std::shared_ptr<TensorBase> output_gradient_;
    bool requires_grad_;

public:
    FunctionNode(std::vector<std::shared_ptr<GraphNode>> inputs)
        : inputs_(inputs) {
        // Check if any input requires gradients
        requires_grad_ = false;
        for (const auto& input : inputs_) {
            if (input->requires_grad()) {
                requires_grad_ = true;
                break;
            }
        }
    }
    
    std::vector<std::shared_ptr<GraphNode>> inputs() const override {
        return inputs_;
    }
    
    void set_gradient(std::shared_ptr<TensorBase> grad) override {
        if (output_gradient_) {
            // Accumulate gradients (add returns unique_ptr)
            auto accumulated = output_gradient_->add(*grad);
            output_gradient_.reset(accumulated.release());
        } else {
            output_gradient_ = grad;
        }
    }
    
    std::shared_ptr<TensorBase> get_gradient() const override {
        return output_gradient_;
    }
    
    bool requires_grad() const override { return requires_grad_; }
    
    std::shared_ptr<TensorBase> get_tensor() const override { return output_tensor_; }

    void clear_gradient() override {
        output_gradient_.reset();
        for (auto& inp : inputs_) {
            inp->clear_gradient();
        }
    }

protected:
    void set_output_tensor(std::shared_ptr<TensorBase> tensor) {
        output_tensor_ = tensor;
    }
};

/**
 * @brief Specific operation nodes
 */

class AddNode : public FunctionNode {
public:
    AddNode(std::shared_ptr<GraphNode> a, std::shared_ptr<GraphNode> b)
        : FunctionNode({a, b}) {}
    
    std::string name() const override { return "Add"; }
    
    void backward() override {
        if (!requires_grad()) return;
        
        auto grad = get_gradient();
        if (!grad) return;
        
        // d(a + b)/da = 1, d(a + b)/db = 1
        // So gradient flows unchanged to both inputs
        if (inputs_[0]->requires_grad()) {
            inputs_[0]->set_gradient(grad);
        }
        if (inputs_[1]->requires_grad()) {
            inputs_[1]->set_gradient(grad);
        }
    }
};

class MulNode : public FunctionNode {
private:
    std::shared_ptr<TensorBase> a_tensor_;
    std::shared_ptr<TensorBase> b_tensor_;

public:
    MulNode(std::shared_ptr<GraphNode> a, std::shared_ptr<GraphNode> b)
        : FunctionNode({a, b}), a_tensor_(a->get_tensor()), b_tensor_(b->get_tensor()) {}
    
    std::string name() const override { return "Mul"; }
    
    void backward() override {
        if (!requires_grad()) return;
        
        auto grad = get_gradient();
        if (!grad) return;
        
        // d(a * b)/da = b, d(a * b)/db = a
        if (inputs_[0]->requires_grad()) {
            auto grad_a = grad->mul(*b_tensor_);
            inputs_[0]->set_gradient(grad_a);
        }
        if (inputs_[1]->requires_grad()) {
            auto grad_b = grad->mul(*a_tensor_);
            inputs_[1]->set_gradient(grad_b);
        }
    }
};

class ScalarMulNode : public FunctionNode {
private:
    double scalar_;

public:
    ScalarMulNode(std::shared_ptr<GraphNode> input, double scalar)
        : FunctionNode({input}), scalar_(scalar) {}
    
    std::string name() const override { return "ScalarMul"; }
    
    void backward() override {
        if (!requires_grad()) return;
        
        auto grad = get_gradient();
        if (!grad) return;
        
        // d(a * scalar)/da = scalar
        if (inputs_[0]->requires_grad()) {
            auto grad_input = grad->scalar_mul(scalar_);
            inputs_[0]->set_gradient(grad_input);
        }
    }
};

class MatMulNode : public FunctionNode {
private:
    std::shared_ptr<TensorBase> a_tensor_;
    std::shared_ptr<TensorBase> b_tensor_;

public:
    MatMulNode(std::shared_ptr<GraphNode> a, std::shared_ptr<GraphNode> b)
        : FunctionNode({a, b}), a_tensor_(a->get_tensor()), b_tensor_(b->get_tensor()) {}
    
    std::string name() const override { return "MatMul"; }
    
    void backward() override {
        if (!requires_grad()) return;
        
        auto grad = get_gradient();
        if (!grad) return;
        
        // For C = A @ B:
        // dC/dA = grad @ B^T
        // dC/dB = A^T @ grad
        
        if (inputs_[0]->requires_grad()) {
            // TODO: Implement transpose and matrix multiply for gradient computation
            // auto grad_a = grad->matmul(b_tensor_->transpose());
            // inputs_[0]->set_gradient(grad_a);
        }
        if (inputs_[1]->requires_grad()) {
            // auto grad_b = a_tensor_->transpose().matmul(*grad);
            // inputs_[1]->set_gradient(grad_b);
        }
    }
};

/**
 * @brief Main computation graph class for executing backward passes
 */
class ComputationGraph {
private:
    std::vector<std::shared_ptr<GraphNode>> topological_order_;
    
public:
    // Execute backward pass from a root node
    void backward(std::shared_ptr<GraphNode> root) {
        // Always add unit gradient to the root (accumulates internally)
        auto ones = create_ones_like(root->get_tensor());
        root->set_gradient(ones);
        
        // Get topological ordering
        std::vector<std::shared_ptr<GraphNode>> topo_order;
        std::unordered_set<GraphNode*> visited;
        topological_sort(root, visited, topo_order);
        
        // Execute backward pass in reverse topological order
        for (auto it = topo_order.rbegin(); it != topo_order.rend(); ++it) {
            (*it)->backward();
        }
    }
    
private:
    void topological_sort(std::shared_ptr<GraphNode> node, 
                         std::unordered_set<GraphNode*>& visited,
                         std::vector<std::shared_ptr<GraphNode>>& order) {
        if (visited.find(node.get()) != visited.end()) {
            return;
        }
        
        visited.insert(node.get());
        
        for (const auto& input : node->inputs()) {
            topological_sort(input, visited, order);
        }
        
        order.push_back(node);
    }
    
    std::shared_ptr<TensorBase> create_ones_like(std::shared_ptr<TensorBase> tensor) {
        using TensorPtr = std::shared_ptr<TensorBase>;
        auto shape = tensor->shape();
        auto dtype = tensor->dtype();
        // Create empty tensor with same dtype/shape
        std::unique_ptr<TensorBase> empty = TensorWrapper<float>::create_empty_tensor_by_dtype_enum(shape, dtype);
        TensorPtr ones(empty.release());

        size_t n = tensor->size();
        switch (dtype) {
            case DType::FLOAT32: {
                std::vector<float> host(n, 1.0f);
                ones->copy_from_host_generic(host.data());
                break;
            }
            case DType::FLOAT64: {
                std::vector<double> host(n, 1.0);
                ones->copy_from_host_generic(host.data());
                break;
            }
            default:
                throw std::runtime_error("create_ones_like: dtype not supported");
        }
        return ones;
    }
};

/**
 * @brief Context manager for gradient computation
 */
class GradContext {
private:
    static thread_local bool grad_enabled_;

public:
    static bool is_grad_enabled() { return grad_enabled_; }
    static void set_grad_enabled(bool enabled) { grad_enabled_ = enabled; }
    
    // RAII class for no_grad contexts
    class NoGradGuard {
        bool prev_state_;
    public:
        NoGradGuard() : prev_state_(grad_enabled_) {
            grad_enabled_ = false;
        }
        ~NoGradGuard() {
            grad_enabled_ = prev_state_;
        }
    };
};

// Initialize thread_local variable
thread_local bool GradContext::grad_enabled_ = true;

#endif // COMPUTATION_GRAPH_H 