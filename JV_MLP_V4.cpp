#include<iostream>
#include<fstream>
#include<vector>
#include<algorithm>
#include<execution>
#include<cassert>
#include<string>
#include<random>
#include<cmath>
#include<algorithm>
#include<typeinfo>
#include<chrono>
#include<sstream>


//#include<bits/stdc++.h>

using namespace std;

/*
Features:

    * Flexibility : Num layers,layer sizes, activations (Linear, Sigmoid, ReLu, Tanh), Optimisers (Adam, Nadam, RMSProp).
    * Numerical Stability.
    * Repeatability of results.

*/


// Fundamental Matrix Operations : Matrix Multiplications, Additions, Addition with scalar and scaling.
/**********************************************************************************/

// Function to multiply A[cur_row:cur_row+batch_size][k] with B[k][m] and return the result.
template <typename T> //works only for numeric datatypes
vector<vector<T>> Mul(vector<vector<T> > &A,vector<vector<T>> &B, int cur_row, int batch_size = 1)
/*
Description:
    performs A[cur_row:cur_row+batch_size] * B
*/
{
    assert(cur_row<A.size()); //sanity check

    auto n = cur_row + batch_size;
    auto k = A[0].size();
    auto m = B[0].size();

    if(n>A.size())
        n = A.size();

    vector<vector<T>> C(batch_size,vector<T>(m,0.0));

    for(auto ii=cur_row;ii<n;ii++)
        for(auto xx=0;xx<k;xx++) 
            for(auto jj=0;jj<m;jj++)
                C[ii-cur_row][jj] += A[ii][xx]*B[xx][jj];
    return C;
}


// Function to add or subtract two vectors
// subtraction, if multiplier is -1
template <typename T> //works only for numeric datatypes
void Add(vector<T> &A,vector<T> &B,int multiplier = 1)
/*
Description:
    performs A + multiplier*B
*/
{
    assert(A.size() == B.size());
    for(auto ii=0;ii<A.size();ii++)
        A[ii] += multiplier*B[ii];
}

// Function to add or subtract two matrices [overloaded]
// Subtraction if multiplier is -1
template <typename T> //works only for numeric datatypes
void Add(vector<vector<T>> &A,vector<vector<T>> &B,int multiplier = 1)
/*
Description:
    performs A + multiplier*B;
*/
{

    assert(A.size() == B.size());//sanity check
    assert(A[0].size() == B[0].size());

    auto n = A.size();
    
    for(auto ii=0;ii<n;ii++)
        for(auto jj=0;jj<A[ii].size();jj++)
                A[ii][jj] += multiplier*B[ii][jj];
}

// Function to add or subtract two tensors [overloaded]
//if multiplier is -1, it becomes matrix subtraction.
template <typename T> //works only for numeric datatypes
void Add(vector<vector<vector<T>>> &A,vector<vector<vector<T>>> &B,int multiplier = 1)
/*
Description:
    performs A + multiplier*B;
*/
{

    assert(A.size() == B.size());
    
    auto n = A.size();
    
    for(auto ii=0;ii<n;ii++)
    {
        assert(A[ii].size() == B[ii].size());
        for(auto jj=0;jj<A[ii].size();jj++)
            for(auto kk=0;kk<A[ii][jj].size();kk++)
                A[ii][jj][kk] += multiplier*B[ii][jj][kk];
    }
}

// Function to add a vector V2 to every row of matrix V1
template <typename T> //works only for numeric datatypes
void Add(vector<vector<T>> &v1,vector<T> &v2)
/*
Description:
    performs A + multiplier*B;
*/
{   
    for(auto &row : v1)
        transform(row.begin(),row.end(),v2.begin(),row.begin(), plus<T>());
}

// Function to add constant to all values of a vector.
template <typename T> //works only for numeric datatypes
void Add(vector<T> &V,T val)
{
        transform(V.begin(),V.end(),V.begin(),
        [val](T & ele) {return ele+ val;});
}

// Function to add constant to all values of a matrix. [overloaded]
template <typename T> //works only for numeric datatypes
void Add(vector<vector<T>> &matrix,T val){
    for(auto &row : matrix){
        transform(row.begin(),row.end(),row.begin(),
        [val](T & ele) {return ele+ val;});
    }
}

// Function to add constant to all values of a Tensor. [overloaded]
template <typename T> //works only for numeric datatypes
void Add(vector<vector<vector<T>>> &tensor,T val){

    for(auto &matrix : tensor)
        for(auto &row : matrix)
            transform(row.begin(),row.end(),row.begin(),
            [val](T & ele) {return ele+ val;});
}



//Function to compute and return c1*v1 + c2*v2, c1 and c2 are scalars, v1&v2 are vectors
vector<double> LinearCombination(vector<double> &v1,double c1,vector<double> &v2,double c2)
{
    assert(v1.size() == v2.size()); //Sanity check

    vector<double> v3(v1.size(),0);

    transform(v1.begin(),v1.end(),v2.begin(),v3.begin(),
    [c1,c2](double v1_ele,double v2_ele){
        return c1*v1_ele + c2*v2_ele;
    });

    return v3;
}

//Function to compute and return c1*v1 + c2*v2, c1 and c2 are scalars, v1&v2 are Matrices [overloaded]
vector<vector<double>> LinearCombination(vector<vector<double>> &v1,double c1,vector<vector<double>> &v2,double c2)
{
    assert(v1.size() == v2.size()); //Sanity check
    assert(v1.size()>0);
    assert(v1[0].size() == v2[0].size());

    /*vector<vector<double>> v3(v1.size(),vector<double>(v1[0].size(),0));

    for(int i=0;i<v1.size();i++)
        transform(v1[i].begin(),v1[i].end(),v2[i].begin(),v3[i].begin(),
        [c1,c2](double v1_ele,double v2_ele){
            return c1*v1_ele + c2*v2_ele;
        });*/

    vector<vector<double>> v3;
    for(int i=0;i<v1.size();i++)
        v3.push_back(LinearCombination(v1[i],c1,v2[i],c2));

    return v3;
}

//Function to compute and return c1*v1 + c2*v2, c1 and c2 are scalars, v1&v2 are TENSORS [overloaded]
vector<vector<vector<double>>> LinearCombination(vector<vector<vector<double>>> &v1,double c1,vector<vector<vector<double>>> &v2,double c2)
{
    assert(v1.size() == v2.size()); //Sanity check
    assert(v1.size()>0);
    
    vector<vector<vector<double>>> v3;

    for(int i=0;i<v1.size();i++)
        v3.push_back(LinearCombination(v1[i],c1,v2[i],c2));

    return v3;
}


//Function to multiply all values in a matrix by a scalar, inplace. [overloading]
template <typename T> //works only for numeric datatypes
vector<T> Scale(vector<T> &v,T val)
{
        vector<T> res(v.size(),0);

        transform(v.begin(),v.end(),res.begin(),
        [val](T & ele) {return ele* val;});

    return res;
}

//Function to multiply all values in a matrix by a scalar, inplace. [overloaded]
template <typename T> //works only for numeric datatypes
void Scale(vector<vector<T>> &matrix,T val){
    for(auto &row : matrix)
        transform(row.begin(),row.end(),row.begin(),
        [val](T & ele) {return ele* val;});
}

//Function to multiply all values in a tensor by a scalar, inplace. [overloaded]
template <typename T> //works only for numeric datatypes
void Scale(vector<vector<vector<T>>> &tensor,T val){

    for(auto &matrix : tensor)
        for(auto &row : matrix)
            transform(row.begin(),row.end(),row.begin(),
            [val](T & ele) {return ele* val;});
}

// Function to compute element wise square Matrix.
template <typename T> //works only for numeric datatypes
void Matrix_sqrt(vector<T> &V)
{
        std::transform(V.begin(), V.end(), V.begin(), [](T val) 
        {
            return std::sqrt(val);
        });
}


// Function to compute element wise square root of a matrix.
template <typename T> //works only for numeric datatypes
void Matrix_sqrt(vector<vector<T>> &matrix)
{
    for (auto& row : matrix) {
        transform(row.begin(), row.end(), row.begin(), [](T val) {
            return std::sqrt(val);
        });
    }
}

// Function to compute element wise square root of a tensor.
template <typename T> //works only for numeric datatypes
void Matrix_sqrt(vector<vector<vector<T>>> &tensor)
{    
    for(auto &matrix : tensor)   
        for (auto& row : matrix)
            transform(row.begin(), row.end(), row.begin(), [](T val) {
                return std::sqrt(val);
            });
}

// Function to compute element wise square Matrix.
template <typename T> //works only for numeric datatypes
void Matrix_square(vector<T> &V)
{
        std::transform(V.begin(), V.end(), V.begin(), [](T val) 
        {
            return val*val;
        });
}


// Function to compute element wise square root of a matrix.
template <typename T> //works only for numeric datatypes
void Matrix_square(vector<vector<T>> &matrix)
{
    for (auto& row : matrix) {
        transform(row.begin(), row.end(), row.begin(), [](T val) {
            return val*val;
        });
    }
}

// Function to compute element wise square root of a tensor.
template <typename T> //works only for numeric datatypes
void Matrix_square(vector<vector<vector<T>>> &tensor)
{    
    for(auto &matrix : tensor)   
        for (auto& row : matrix)
            transform(row.begin(), row.end(), row.begin(), [](T val) {
                return val*val;
            });
}

//Function to perform element wise division of two vectors
template <typename T> //works only for numeric datatypes
void Matrix_divide(vector<T> &A,vector<T> &B)
{
        std::transform(A.begin(), A.end(), B.begin(), A.begin(), std::divides<T>());
}

//Function to perform element wise division of two matrices [overloaded]
template <typename T> //works only for numeric datatypes
void Matrix_divide(vector<vector<T>> &A,vector<vector<T>> &B)
{
    for (int i = 0; i < A.size(); i++)
        std::transform(A[i].begin(), A[i].end(), B[i].begin(), A[i].begin(), std::divides<T>());
}

//Function to perform element wise division of two tensors [overloaded]
template <typename T> //works only for numeric datatypes
void Matrix_divide(vector<vector<vector<T>>> &A,vector<vector<vector<T>>> &B)
{
    
    for(int i = 0; i < A.size(); i++)
        for(int j = 0; j < A[i].size(); j++)
            std::transform(A[i][j].begin(), A[i][j].end(), B[i][j].begin(), A[i][j].begin(), std::divides<T>());
}

//Function to find and return the index of the largest element in a vector 
template <typename T> //works only for numeric datatypes
T argmax(vector<T> &V)
{
    assert(!V.empty());
    auto max_ele_pos = max_element(V.begin(),V.end());

    return distance(V.begin(),max_ele_pos);
}

//Function to find and return the index of the largest element in a vector [overloaded]
template <typename T> //works only for numeric datatypes
vector<T> argmax(vector<vector<T>> &V)
{
    vector<T> res;

    for(auto &row : V)
    {
        auto max_ele = max_element(row.begin(),row.end());
        auto amax = distance(row.begin(),max_ele);

        res.push_back(amax);
    }

    return res;
}

// to compute the total cross entropy loss between the vectors within a vector
template <typename T> //works only for numeric datatypes
T cross_entropy_loss(vector<vector<T>> &pred_probs, vector<T> &true_probs, int cur_row)
{
    double loss=0;
    for(int i=0;i<pred_probs.size();i++)
        loss = loss + -(log(pred_probs[i][int(true_probs[i+cur_row])]));

    return loss;
}

// function to compute accuracy, given the predicted classes preds and true_labels
// preds must be predicted classes
template <typename T> //works only for numeric datatypes
T compute_accuracy(vector<T> &preds, vector<T> &true_labels, int cur_row = -1)
{
    int c = (cur_row == -1)?0:cur_row;

    double correct_preds = 0;

    for(int i =0 ;i<preds.size();i++)
        if(int(preds[i]) == int(true_labels[i+c]))
            correct_preds++;

    double accuracy = (correct_preds*100)/(double(preds.size()));

    return accuracy;
}

// to match types v1 and v2 are vect of vect, but actually they are just 1D vectors
template <typename T> //works only for numeric datatypes
vector<vector<T>> Outer(vector<vector<T>> &v1, vector<T> &v2)
{
    vector<vector<T>> result(v1[0].size(),vector<T>(v2.size(),0));
    
    for(int i=0;i<v1[0].size();i++)
    {
        auto ele = v1[0][i];
        for(int j=0;j<v2.size();j++)
        {
            result[i][j] += ele*v2[j];
        }
    }
    return result;

}

// To compute the product of a MxN matrix with Nx1 Vector (vector stored as 1xN)
template <typename T> //works only for numeric datatypes
vector<T> MatrixVectorProduct(vector<vector<T>> &M,vector<T> &V)
{
    assert(M[0].size() == V.size());

    vector<T> res(M.size());
    for(int i=0;i<M.size();i++)
        for(int j=0;j<M[0].size();j++)
            res[i] += M[i][j]*V[j];

    return res;

}

// Function to compute element wise products of two vectors
// aka Hadamard Product
template <typename T> //works only for numeric datatypes
vector<T> Hadamard(vector<T> &v1, vector<T> &v2)
{
    assert(v1.size() == v2.size());

    vector<T> v3(v1);
    for(int i=0;i<v1.size();i++)
        v3[i] = v1[i]*v2[i];

    return v3;

}


//function to print elements of a vector
template <typename T>
void print(vector<T> &V)
{
    for(auto &ele : V)
        cout<<ele<<" ";
    cout<<endl;
}

//function to print elements of a matrix of type T [overloaded]
template <typename T> 
void print(vector<vector<T>> &V)
{
    for(auto &row : V)
        print(row);
    cout<<endl;
}

//function to print elements of a tensor of type T [overloaded]
template <typename T> 
void print(vector<vector<vector<T>>> &V)
{
    for(auto &row : V)
        print(row);
    cout<<endl;
}

//function to overwrite all elements of an existing container with 0s
template <typename T> //works only for numeric datatypes
void fill_with_zeros(vector<T> &M)
{
    if(M.size()==0)
        return;

    for(int i=0;i<M.size();i++)
        M[i] = 0;
}

//function to overwrite all elements of an existing container with 0s [overloaded]
template <typename T> //works only for numeric datatypes
void fill_with_zeros(vector<vector<T>> &M)
{
    if(M.size()==0)
        return;
    assert(M[0].size()>0);

    /*for(int i=0;i<M.size();i++)
        for(int j=0;j<M[i].size();j++)
            M[i][j] = 0;*/
    for(int i=0;i<M.size();i++)
        fill_with_zeros(M[i]);  
} 

//function to overwrite all elements of an existing container with 0s [overloaded]
template <typename T> //works only for numeric datatypes
void fill_with_zeros(vector<vector<vector<T>>> &M)
{
    if(M.size()==0)
        return;

    assert(M[0].size()>0);
    assert(M[0][0].size()>0);

    /*for(int i=0;i<M.size();i++)
        for(int j=0;j<M[i].size();j++)
            for(int k=0;k<M[i][j].size();k++)
            M[i][j][k] = 0;*/

    for(int i=0;i<M.size();i++)
        fill_with_zeros(M[i]);
}


/**********************************************************************************/

/*
Layers can be vector<vector<vector<double>>>
Inputs (may be to a constructor) : num layers (int), an array of layer sizes, initialization mech.
Activation and Optimisers can be separate classes

Network class has 
    constructor to create network and initialize the weights.
    Forward Pass.
    Bacward Pass.



*/

// Class that defines the structure of a layer in a neural network.
class Layer
{
    public:
        int size,inp_dim,out_dim;
        vector<vector<double>> weights;
        vector<double> biases;

        Layer(int neurons,int in_dim,int op_dim)
        {
            size = neurons;
            inp_dim = in_dim;
            out_dim = op_dim;
            biases.assign(op_dim,0.01); //Initialize biases to 0.01, a small non zero value.

        }

        //Function to perform xavier_normal initialization.
        void xavier_normal_initialization(int seed = 76)
        {

            int fan_in = inp_dim, fan_out = out_dim;
            double mu = 0; // mean is zero for
            double sigma = sqrt(2.0/(fan_in+fan_out));

            mt19937 gen(seed); //pseudo random number generator, with a seed.
            normal_distribution<double> N(mu,sigma); //mu passed would be 0 and sigma would be as per xavier initialization mech.

            weights.resize(fan_in,vector<double>(fan_out,0)); //Fixing the shape of the weights matrix and initializing with 0s.

            for(int i=0;i<fan_in;i++)
            {
                for(int j=0;j<fan_out;j++)
                    weights[i][j] = N(gen);

            }

        }
};

// Class to support activation functions : to apply after neural network layers.
// Supports : Tanh, ReLU, Softmax

class Activation
{
    public:

        //tanh for batch size = 1
        void Tanh(vector<double> &W)
        {
                transform(W.begin(),W.end(),W.begin(),[](double &element)
                {
                    return tanh(element);   
                });
        }

        //tanh for batch size>1 [overloading]
        void Tanh(vector<vector<double>> &W)
        {
                for(auto &row : W)
                    transform(row.begin(),row.end(),row.begin(),[](double &element)
                    {
                        return tanh(element);   
                    });
        }

        //ReLU for batch size=1
        void ReLU(vector<double> &W)
        {

            auto relu = [](double& element) -> double {
                return (element>0)?element:0;
            };

            transform(W.begin(),W.end(),W.begin(),relu);

        }

        //ReLU for batch size>1 [overloading]
        void ReLU(vector<vector<double>> &W)
        {

            auto relu = [](double& element) -> double {
                return (element>0)?element:0;
            };

            for(auto &row : W)
                transform(row.begin(),row.end(),row.begin(),relu);
        }



        //function to compute the softmax over a vector, with Batch size = 1
        void Softmax(vector<double> &W)
        {
            double sum = 0;//, max = *max_element(W.begin(),W.end());

            for(int i=0;i<W.size();i++)
            {
                W[i] = exp(W[i]);
                sum += W[i];
            }

            for(int i=0;i<W.size();i++)
                W[i]/=sum;
        }

        //function to compute the softmax over a vector, with Batch size>1 [overloading]
        void Softmax(vector<vector<double>> &W)
        {
            for(auto &row : W)
            {
                double sum = 0;//,max = *max_element(row.begin(),row.end());
                for(int i=0;i<row.size();i++)
                {
                    row[i] = exp(row[i]);
                    sum += row[i];
                }

                for(int i=0;i<row.size();i++)
                    row[i]/=sum;
            }
        }
};

//Class to support functions for the gradient of activation functions
class GradActivation
{
    public:
        //gradient of tanh for batch size = 1
        void Tanh_d(vector<double> &W)
        {
                transform(W.begin(),W.end(),W.begin(),[](double &element)
                {
                    return 1-tanh(element)*tanh(element);
                });
        }

        //gradient of tanh for batch size>1 [overloading]
        void Tanh_d(vector<vector<double>> &W)
        {
                for(auto &row : W)
                    transform(row.begin(),row.end(),row.begin(),[](double &element)
                    {
                        return 1-tanh(element)*tanh(element);
                    });
        }

        //gradient of ReLU for batch size=1
        void ReLU_d(vector<double> &W)
        {

            auto relu = [](double& element) -> double 
            {
                return (element>0)?1:0;
            };

            transform(W.begin(),W.end(),W.begin(),relu);

        }

        //gradient of ReLU for batch size>1 [overloading]
        void ReLU_d(vector<vector<double>> &W)
        {

            auto relu = [](double& element) -> double {
                return (element>0)?1:0;
            };

            for(auto &row : W)
                transform(row.begin(),row.end(),row.begin(),relu);
        }

};

// Class to define a NNet model, with supporting methods like forward, backward passes and train functions.
class Network
{

        public:

            string hidden_activation; //activation applied after hidden layer.
            string output_activation; //output activation.
            vector<Layer> layers;     //A vector of layer class objects

            // The constructor of the class : To initialize hyperparameters, create and initialize layers. 
            Network(vector<int> &hidden_sizes,int input_size,int output_size,string h_acti,string op_acti)
            {
                
                assert(!hidden_sizes.empty());

                hidden_activation = h_acti;
                output_activation = op_acti;

                createLayers(hidden_sizes,input_size,output_size);

            }

            // Function to create the layers of the neural network, based on the input, output sizes and hidden_sizes vector.
            void createLayers(vector<int> &hidden_sizes, int input_size, int output_size)
            {
                for(int i = 0;i<hidden_sizes.size();i++)
                {

                    if(i==0) // The First hidden layer
                    {
                        Layer layer(hidden_sizes[i],input_size,hidden_sizes[i]);
                        layers.push_back(layer);
                    }
                    else // All (but last) hidden layers
                    {
                        Layer layer(hidden_sizes[i],hidden_sizes[i-1],hidden_sizes[i]);
                        layers.push_back(layer);
                    }

                }

                Layer layer(output_size,hidden_sizes.back(),output_size); //last hidden layer
                layers.push_back(layer);
            }

            // Function to initialize the weights of the network
            // Weights are initialized using Xavier initialization and biases are initialized to zeros.
            void InitializeWeights(int seed = 76)
            {
                for(auto &layer : layers)
                    layer.xavier_normal_initialization(seed);
            }

            // Function to perform the forward pass/propagation
            // As of now works for batch_size=1 only [can be updagraded]
            // It stores the states i.e pre-activation outputs of all hidden layers in "all_Ais"
            // and post-activation outputs of all hidden layers and output in "all_His"
            // returns the predicted probabilities
            vector<vector<double>> forward(vector<vector<double>> &data,vector<vector<vector<double>>> &all_Ais,vector<vector<vector<double>>> &all_His, int current_row, Activation &acti, int batch_size = 1, int test_mode = 0)
            {

                //to make sure older data (if any) is cleared.
                all_Ais.clear();
                all_His.clear();

                vector<vector<double>> Ai;
                
                //this assumes that batch size is restricted to 1.
                auto data_row = data[current_row];

                if(!test_mode) //no need to store state vectors for inference
                {
                    vector<vector<double>> cur_data_row;
                    cur_data_row.push_back(data_row);
                    all_His.push_back(cur_data_row); //all_His[0] should be the relevant input data
                }

                for(int i =0; i<layers.size(); i++)
                {
                    if(i == 1) // from the 2nd iteration onwards we work with Activation not input data.
                        current_row = 0;

                    if(i==0)
                        Ai = Mul(data,layers[i].weights,current_row,batch_size);
                    else
                        Ai = Mul(Ai,layers[i].weights,current_row,batch_size);
                    
                    Add(Ai,layers[i].biases); // xTw + b transformation

                    if(!test_mode)
                        all_Ais.push_back(Ai);

                    if(i<layers.size()-1) //for all hidden layers
                    {
                        if(hidden_activation == "tanh")
                            acti.Tanh(Ai);
                        else if(hidden_activation == "relu")
                            acti.ReLU(Ai);
                    }
                    else //for output layer
                        acti.Softmax(Ai);

                    
                    if(!test_mode)
                        all_His.push_back(Ai);

                }

                return Ai; //return prediction probabilities
                
            }

            // Performs the backward propagation algorithm over the neural net
            // uses the state vectors "all_Ais" and "all_His" from forward pass
            // and stores gradients in "dw_list" and "db_list"
            // due to the reverse order, the dw_list[0] would correspond to gradients of output layer and so on.
            // NOTE : Auto-differentiation is not used, instead we limit to problem of multi-class classification [with cross entropy loss]
            // and using closed form expressions of the gradients and the activations
            // currently supported activations : tanh, ReLU, softmax
            void backprop(vector<vector<vector<double>>> &all_Ais,vector<vector<vector<double>>> &all_His,double y_true,vector<double> &y_pred, vector<vector<vector<double>>> &dw_list,vector<vector<double>> &db_list)
            {

                GradActivation activ_grad;

                dw_list.clear(); //to make sure any older data is removed.
                db_list.clear();

                //assuming batch size = 1
                auto grad_aL = y_pred;
                
                grad_aL[int(y_true)] = y_pred[int(y_true)]-1;

                auto grad_ai = grad_aL;

                for(int i=layers.size()-1;i>=0;i--) //iterate over each hidden layer backwards : backprop
                {
                    dw_list.push_back(Outer(all_His[i],grad_ai));
                    db_list.push_back(grad_ai); //gradient of biases = grad_ai

                    if(i>0) // if a previous layer exists, back propagate.
                    {
                        vector<double> grad_h_prev = MatrixVectorProduct(layers[i].weights,grad_ai);
                        vector<double> grad_a_prev(grad_h_prev.size()); //dimension of hi=ai always
                        
                        auto prev_layer_ai_grad = all_Ais[i-1];

                        if(hidden_activation == "tanh")
                            activ_grad.Tanh_d(prev_layer_ai_grad);

                        else if(hidden_activation == "relu")
                            activ_grad.ReLU_d(prev_layer_ai_grad);

                        grad_a_prev = Hadamard(grad_h_prev , prev_layer_ai_grad[0]); //prev_layer_ai_grad is a vector of 1 vector, we just need a vector;
                        grad_ai = grad_a_prev;
                    }
                }
            }

            // Function to print the weights of a given layer l of the network
            void PrintWeights(int l)
            {
                for(int i=0;i<layers[l].inp_dim;i++)
                {
                    for(int j=0;j<layers[l].out_dim;j++)
                        cout<<layers[l].weights[i][j]<<" ";
                    cout<<"\n";
                }
                cout<<endl;
            }


            // Function to print the model architecture.
            void printArchitecture()
            {
                cout<<"\n============================================ Network Architecture ============================================\n\n";
                cout<<"Number of Layers : "<<layers.size()<<endl<<endl;
                
                cout<<"Layer : 0(inp)\t Neurons : "<<layers[0].inp_dim<<"\t Dim : NA\t Params : NA\n\n";
                
                long long total_params = 0;
                for(int i =0;i<layers.size();i++)
                {
                    long long params = 0;
                    params += layers[i].inp_dim*layers[i].out_dim; //weights matrix
                    params += layers[i].out_dim; //biases
                    total_params += params;

                    cout<<"Layer : "<<i+1<<"\t Neurons : "<<layers[i].size<<"\t Dim : "<<layers[i].inp_dim<<"x"<<layers[i].out_dim<<"\t Params : "<<params<<endl;
                    
                    if(i < layers.size()-1)
                        cout<<hidden_activation<<" Activation\n"<<endl;
                    else
                        cout<<output_activation<<" Activation\n"<<endl;
                }
                cout<<"Total Params : "<<total_params<<endl;
                cout<<"\n==============================================================================================================\n";
            }

            //function to print the dimensions of the weights of all layers
            void printWeightDims()
            {
                for(int i=0;i<layers.size();i++)
                {
                    cout<<"\t\tCur layer("<<i<<") Weights dim : "<<layers[i].weights.size()<<"x"<<layers[i].weights[0].size()<<"\n";
                }
            }

            //function to print the dimensions of any matrix stored as a vector<vector<double>>
            void printDims(vector<vector<double>> &V)
            {
                for(auto &row : V)
                {
                    cout<<"\t\tdim : "<<row.size()<<"\n";
                }
            }

            //function to print the dimensions of any matrix stored as a vector<vector<vector<double>>>
            void printDims(vector<vector<vector<double>>> &V)
            {
                for(int i=0;i<V.size();i++)
                        cout<<"\t\tdim : "<<V[i].size()<<"x"<<V[i][0].size()<<"\n";
            }

};


// Class to implement the optimization algorithms, for training the neural net
// Supports  : Stochastic and Batch GD, Momentum based GD, RMSprop and Adam.
class Optimiser
{
    public:

        //Function to gather the predictions of the model on the data (X,Y) and print the stats (accuracy and loss).
        void evaluateModel(Network &model,vector<vector<double>> &X,vector<double> &Y,string dtype, bool print_stats = true, int test_mode = 1)
        {

            double correct_preds = 0;
            double avg_loss = 0;

            Activation activ;
            vector<vector<vector<double>>> all_Ais;//all_Ais and all_His are not needed actually, but creating for the sake of matching function call parameter sequence
            vector<vector<vector<double>>> all_His;

            for(int temp = 0;temp<X.size();temp++)
            {
                auto current_row = temp;
                int batch_size = 1;
                auto y_pred_probs = model.forward(X,all_Ais,all_His, current_row, activ, batch_size,test_mode); //forward pass works on one sample at time
                double y_true = Y[current_row];

                //to compute accuracy later.
                auto cur_prediction = argmax(y_pred_probs);
                if(int(cur_prediction[0]) == int(Y[current_row]))
                    correct_preds++;

                avg_loss += cross_entropy_loss(y_pred_probs,Y,current_row);

            }

            double accuracy = correct_preds*100/X.size();
            cout<<dtype<<" Acc. : "<<accuracy<<"%\tAvg. "<<dtype<<" Loss : "<<avg_loss/X.size()<<endl;

        }

        // Function to implement the stochastic gradient descent algorithm.
        void sgd(vector<vector<double>> &X_train,vector<double> &Y_train,vector<vector<double>> &X_valid,vector<double> &Y_valid,Network &model,double lr,int epochs,int batch_size,double l2_param)
        {
            Activation activ; //creating an instance of the activation class

            vector<vector<vector<double>>> all_Ais;
            vector<vector<vector<double>>> all_His;

            vector<vector<vector<double>>> dw;
            vector<vector<double>> db;

            //initialize grads of weights and biases to zeros.
            for(int i = 0;i<model.layers.size();i++)
            {
                vector<vector<double>> tmp_w(model.layers[i].inp_dim,vector<double>(model.layers[i].out_dim,0));
                dw.push_back(tmp_w);

                vector<double> tmp_b(model.layers[i].out_dim,0);
                db.push_back(tmp_b);
            }

            vector<vector<vector<double>>> dw_cur(dw);
            vector<vector<double>> db_cur(db);

            for(int epoch = 0;epoch<epochs;epoch++)
            {

                auto start = chrono::high_resolution_clock::now();
                cout<<"Epoch "<<epoch+1<<flush;
                int current_row  = 0;
                double correct_preds = 0;
                double avg_loss = 0;
                vector<double> preds;

                for(int zz=0;zz<X_train.size();zz++) //just a dummy variable to loop over the train data
                {
                    auto y_pred_probs = model.forward(X_train,all_Ais,all_His, current_row, activ, 1); //forward pass works on one sample at time
                    double y_true = Y_train[current_row];
                    //to compute accuracy later.
                    auto cur_prediction = argmax(y_pred_probs[0]);
                    
                    if(int(cur_prediction) == int(Y_train[current_row]))
                        correct_preds++;
                    avg_loss += cross_entropy_loss(y_pred_probs,Y_train,current_row);

                    model.backprop(all_Ais,all_His,y_true,y_pred_probs[0], dw_cur,db_cur);

                    //backprop reverses the order of weight matrices, undo it
                    reverse(dw_cur.begin(),dw_cur.end());
                    reverse(db_cur.begin(),db_cur.end());

                    Add(dw,dw_cur);
                    Add(db,db_cur);
                    
                    current_row++;

                    if(current_row%batch_size == 0)
                    {
                        for(int l=0;l<model.layers.size();l++)
                        {
                            Scale(dw[l],lr*(1+l2_param));
                            Add(model.layers[l].weights,dw[l],-1);

                            Scale(db[l],lr*(1+l2_param));
                            Add(model.layers[l].biases,db[l],-1);
                        }

                        fill_with_zeros(dw);
                        fill_with_zeros(db);
                    }
                
                }

                double accuracy = correct_preds*100/X_train.size();

                auto end = chrono::high_resolution_clock::now();

                chrono::duration<double> time = end-start;

                cout<<"\tTrain Acc. : "<<accuracy<<"%\tAvg. Train Loss : "<<avg_loss/X_train.size()<<"\t Time : "<<time.count()<<"s"<<" "<<flush;

                bool print_stats = true;
                int test_mode = 1;
                evaluateModel(model,X_valid,Y_valid,"\tVal",print_stats,test_mode);


            }


        }

        // Function to implement the Momentum based stochastic gradient descent algorithm.
        void gd_momentum(vector<vector<double>> &X_train,vector<double> &Y_train,vector<vector<double>> &X_valid,vector<double> &Y_valid,Network &model,double lr,int epochs,int batch_size,double l2_param)
        {
            
            double momentum = 0.9; //momentum parameter
            
            Activation activ; //creating an instance of the activation class

            vector<vector<vector<double>>> all_Ais;
            vector<vector<vector<double>>> all_His;

            vector<vector<vector<double>>> dw;
            vector<vector<double>> db;

            //initialize grads of weights and biases to zeros.
            for(int i = 0;i<model.layers.size();i++)
            {
                vector<vector<double>> tmp_w(model.layers[i].inp_dim,vector<double>(model.layers[i].out_dim,0));
                dw.push_back(tmp_w);

                vector<double> tmp_b(model.layers[i].out_dim,0);
                db.push_back(tmp_b);
            }

            vector<vector<vector<double>>> dw_cur(dw);
            vector<vector<double>> db_cur(db);

            //for momentum
            vector<vector<vector<double>>> prev_uw(dw);
            vector<vector<double>> prev_ub(db);

            for(int epoch = 0;epoch<epochs;epoch++)
            {

                auto start = chrono::high_resolution_clock::now();
                cout<<"Epoch "<<epoch+1<<flush;
                int current_row  = 0;
                double correct_preds = 0;
                double avg_loss = 0;
                vector<double> preds;

                for(int zz=0;zz<X_train.size();zz++) //just a dummy variable to loop over the train data
                {
                    auto y_pred_probs = model.forward(X_train,all_Ais,all_His, current_row, activ, 1); //forward pass works on one sample at time
                    double y_true = Y_train[current_row];

                    //to compute accuracy later.
                    auto cur_prediction = argmax(y_pred_probs[0]);
                    //preds.push_back(cur_prediction[0]);

                    if(int(cur_prediction) == int(Y_train[current_row]))
                        correct_preds++;

                    avg_loss += cross_entropy_loss(y_pred_probs,Y_train,current_row);

                    model.backprop(all_Ais,all_His,y_true,y_pred_probs[0], dw_cur,db_cur);

                    //backprop reverses the order of weight matrices, undo it
                    reverse(dw_cur.begin(),dw_cur.end());
                    reverse(db_cur.begin(),db_cur.end());

                    Add(dw,dw_cur);
                    Add(db,db_cur);
                    
                    current_row++;

                    if(current_row%batch_size == 0)
                    {
                        auto uw = LinearCombination(prev_uw,momentum,dw,lr);
                        auto ub = LinearCombination(prev_ub,momentum,db,lr);

                        auto temp_uw = LinearCombination(uw,1,dw,lr*l2_param);
                        auto temp_ub = LinearCombination(ub,1,db,lr*l2_param);

                        for(int l=0;l<model.layers.size();l++)
                        {
                            Add(model.layers[l].weights,temp_uw[l],-1);
                            Add(model.layers[l].biases,temp_ub[l],-1);
                        }

                        prev_uw = uw;
                        prev_ub = ub;

                        fill_with_zeros(dw);
                        fill_with_zeros(db);
                    }
                
                }

                double accuracy = correct_preds*100/X_train.size();
                auto end = chrono::high_resolution_clock::now();
                chrono::duration<double> time = end-start;
                cout<<"\tTrain Acc. : "<<accuracy<<"%\tAvg. Train Loss : "<<avg_loss/X_train.size()<<"\t Time : "<<time.count()<<"s"<<" "<<flush;
                
                bool print_stats = true;
                int test_mode = 1;
                evaluateModel(model,X_valid,Y_valid,"\tVal",print_stats,test_mode);

            }


        }

        // Function to implement the RMSprop algorithm.
        void rmsprop(vector<vector<double>> &X_train,vector<double> &Y_train,vector<vector<double>> &X_valid,vector<double> &Y_valid,Network &model,double lr,int epochs,int batch_size,double l2_param)
        {
            
            double beta = 0.5; //algorithm parameter
            double epsilon = 1e-4; //algorithm parameter
            
            Activation activ; //creating an instance of the activation class

            vector<vector<vector<double>>> all_Ais;
            vector<vector<vector<double>>> all_His;

            vector<vector<vector<double>>> dw;
            vector<vector<double>> db;

            //initialize grads of weights and biases to zeros.
            for(int i = 0;i<model.layers.size();i++)
            {
                vector<vector<double>> tmp_w(model.layers[i].inp_dim,vector<double>(model.layers[i].out_dim,0));
                dw.push_back(tmp_w);

                vector<double> tmp_b(model.layers[i].out_dim,0);
                db.push_back(tmp_b);
            }

            vector<vector<vector<double>>> dw_cur(dw);
            vector<vector<double>> db_cur(db);
            vector<vector<vector<double>>> v_w(dw);
            vector<vector<double>> v_b(db);

            for(int epoch = 0;epoch<epochs;epoch++)
            {

                auto start = chrono::high_resolution_clock::now();
                cout<<"Epoch "<<epoch+1<<flush;
                int current_row  = 0;
                double correct_preds = 0;
                double avg_loss = 0;
                vector<double> preds;

                for(int zz=0;zz<X_train.size();zz++) //just a dummy variable to loop over the train data
                {
                    auto y_pred_probs = model.forward(X_train,all_Ais,all_His, current_row, activ, 1); //forward pass works on one sample at time
                    double y_true = Y_train[current_row];
                    auto cur_prediction = argmax(y_pred_probs[0]);//to compute accuracy later.
                    
                    if(int(cur_prediction) == int(Y_train[current_row]))
                        correct_preds++;
                    avg_loss += cross_entropy_loss(y_pred_probs,Y_train,current_row);

                    model.backprop(all_Ais,all_His,y_true,y_pred_probs[0], dw_cur,db_cur);

                    //backprop reverses the order of weight matrices, undo it
                    reverse(dw_cur.begin(),dw_cur.end());
                    reverse(db_cur.begin(),db_cur.end());

                    Add(dw,dw_cur);
                    Add(db,db_cur);
                    
                    current_row++;

                    if(current_row%batch_size == 0)
                    {
                        auto dw_tmp = dw;
                        auto db_tmp = db;

                        Matrix_square(dw_tmp);
                        Matrix_square(db_tmp);

                        v_w = LinearCombination(v_w,beta,dw_tmp,1-beta);
                        v_b = LinearCombination(v_b,beta,db_tmp,1-beta);
                        
                        auto vw_denominator = v_w;
                        auto vb_denominator = v_b;

                        Matrix_sqrt(vw_denominator);
                        Matrix_sqrt(vb_denominator);

                        Add(vw_denominator,epsilon);
                        Add(vb_denominator,epsilon);

                        dw_tmp = dw;
                        db_tmp = db;

                        Matrix_divide(dw_tmp,vw_denominator);
                        Matrix_divide(db_tmp,vb_denominator);

                        dw_tmp = LinearCombination(dw_tmp,lr,dw,lr*l2_param);
                        db_tmp = LinearCombination(db_tmp,lr,db,lr*l2_param);

                        for(int l=0;l<model.layers.size();l++)
                        {
                            Add(model.layers[l].weights,dw_tmp[l],-1);
                            Add(model.layers[l].biases,db_tmp[l],-1);
                        }

                        fill_with_zeros(dw);
                        fill_with_zeros(db);
                    }
                
                }

                double accuracy = correct_preds*100/X_train.size();
                auto end = chrono::high_resolution_clock::now();
                chrono::duration<double> time = end-start;
                
                cout<<"\tTrain Acc. : "<<accuracy<<"%\tAvg. Train Loss : "<<avg_loss/X_train.size()<<"\t Time : "<<time.count()<<"s"<<" "<<flush;
                bool print_stats = true;
                int test_mode = 1;
                evaluateModel(model,X_valid,Y_valid,"\tVal",print_stats,test_mode);

            }
        }

        // Function to implemet the Adam [ADAptive Moments] algorithm.
        void adam(vector<vector<double>> &X_train,vector<double> &Y_train,vector<vector<double>> &X_valid,vector<double> &Y_valid,Network &model,double lr,int epochs,int batch_size,double l2_param)
        {
            
            double beta1 = 0.9,beta2 = 0.999,epsilon=1e-8; //algorithm parameters
            int update_count= 0;

            Activation activ; //creating an instance of the activation class

            vector<vector<vector<double>>> all_Ais;
            vector<vector<vector<double>>> all_His;

            vector<vector<vector<double>>> dw;
            vector<vector<double>> db;

            //initialize grads of weights and biases to zeros.
            for(int i = 0;i<model.layers.size();i++)
            {
                vector<vector<double>> tmp_w(model.layers[i].inp_dim,vector<double>(model.layers[i].out_dim,0));
                dw.push_back(tmp_w);

                vector<double> tmp_b(model.layers[i].out_dim,0);
                db.push_back(tmp_b);
            }

            vector<vector<vector<double>>> dw_cur(dw);
            vector<vector<double>> db_cur(db);

            vector<vector<vector<double>>> v_w(dw);
            vector<vector<double>> v_b(db);
            vector<vector<vector<double>>> m_w(dw);
            vector<vector<double>> m_b(db);

            for(int epoch = 0;epoch<epochs;epoch++)
            {

                auto start = chrono::high_resolution_clock::now();
                cout<<"Epoch "<<epoch+1<<flush;
                int current_row  = 0;
                double correct_preds = 0;
                double avg_loss = 0;
                vector<double> preds;

                for(int zz=0;zz<X_train.size();zz++) //just a dummy variable to loop over the train data
                {
                    auto y_pred_probs = model.forward(X_train,all_Ais,all_His, current_row, activ, 1); //forward pass works on one sample at time
                    double y_true = Y_train[current_row];
                    auto cur_prediction = argmax(y_pred_probs[0]);//to compute accuracy later.
                    
                    if(int(cur_prediction) == int(Y_train[current_row]))
                        correct_preds++;
                    avg_loss += cross_entropy_loss(y_pred_probs,Y_train,current_row);

                    model.backprop(all_Ais,all_His,y_true,y_pred_probs[0], dw_cur,db_cur);

                    //backprop reverses the order of weight matrices, undo it
                    reverse(dw_cur.begin(),dw_cur.end());
                    reverse(db_cur.begin(),db_cur.end());

                    Add(dw,dw_cur);
                    Add(db,db_cur);
                    
                    current_row++;

                    if(current_row%batch_size == 0)
                    {
                        update_count++;
                        
                        auto dw_tmp = dw;
                        auto db_tmp = db;

                        Matrix_square(dw_tmp);
                        Matrix_square(db_tmp);
                        
                        m_w = LinearCombination(m_w,beta1,dw,1-beta1);
                        m_b = LinearCombination(m_b,beta1,db,1-beta1);
                        
                        v_w = LinearCombination(v_w,beta2,dw_tmp,1-beta2);
                        v_b = LinearCombination(v_b,beta2,db_tmp,1-beta2);

                        auto mw_hat = m_w;
                        auto mb_hat = m_b;
                        auto vw_hat = v_w;
                        auto vb_hat = v_b;

                        Scale(mw_hat,1-pow(beta1,update_count));
                        Scale(mb_hat,1-pow(beta1,update_count));
                        Scale(vw_hat,1-pow(beta2,update_count));
                        Scale(vb_hat,1-pow(beta2,update_count));
                        
                        auto vw_denominator_hat = vw_hat;
                        auto vb_denominator_hat = vb_hat;

                        Matrix_sqrt(vw_denominator_hat);
                        Matrix_sqrt(vb_denominator_hat);

                        Add(vw_denominator_hat,epsilon);
                        Add(vb_denominator_hat,epsilon);

                        dw_tmp = mw_hat;
                        db_tmp = mb_hat;

                        Matrix_divide(dw_tmp,vw_denominator_hat);
                        Matrix_divide(db_tmp,vb_denominator_hat);

                        dw_tmp = LinearCombination(dw_tmp,lr,dw,lr*l2_param);
                        db_tmp = LinearCombination(db_tmp,lr,db,lr*l2_param);

                        for(int l=0;l<model.layers.size();l++)
                        {
                            Add(model.layers[l].weights,dw_tmp[l],-1);
                            Add(model.layers[l].biases,db_tmp[l],-1);
                        }

                        fill_with_zeros(dw);
                        fill_with_zeros(db);
                    }
                
                }

                double accuracy = correct_preds*100/X_train.size();
                auto end = chrono::high_resolution_clock::now();
                chrono::duration<double> time = end-start;
                
                cout<<"\tTrain Acc. : "<<accuracy<<"%\tAvg. Train Loss : "<<avg_loss/X_train.size()<<"\t Time : "<<time.count()<<"s"<<" "<<flush;
                bool print_stats = true;
                int test_mode = 1;
                evaluateModel(model,X_valid,Y_valid,"\tVal",print_stats,test_mode);

            }


        }

};

// A namespace for the functions that read the images and lables from files.
namespace input{

    vector<vector<double>> readImages(string filename)
    {
        ifstream file(filename);
        string line;
        vector<vector<double>> array;
        vector<double> row;

        while (getline(file, line)) 
        {
            stringstream ss(line);
            vector<double> row;
            double value;

            while (ss >> value) 
                row.push_back(value);

            array.push_back(row);
        }

        return array;
    }

    vector<double> readLabels(string filename)
    {
        ifstream file(filename);
        vector<double> array;
        double value;

        // Read values from the file and store them in the vector
        while (file >> value) 
            array.push_back(value);
        
        return array;

    }
    
}


int main() {

    //reading Train,Validation and Test Images and Labels
    cout<<"Reading Data...\n";
    auto X_train = input::readImages("X_train.txt");
    auto Y_train = input::readLabels("Y_train.txt");
    auto X_valid = input::readImages("X_valid.txt");
    auto Y_valid = input::readLabels("Y_valid.txt");
    auto X_test = input::readImages("X_test.txt");
    auto Y_test = input::readLabels("Y_test.txt");

    cout<<"\n Data Reading Complete...\n";
    cout<<"\n\n\tTrain Data Size \t"<<X_train.size()<<"x"<<X_train[0].size()<<endl;
    cout<<"\tTrain Labels Size   \t"<<Y_train.size()<<endl;
    cout<<"\n\tValidation Data Size\t"<<X_valid.size()<<"x"<<X_valid[0].size()<<endl;
    cout<<"\tValidation Labels Size\t"<<Y_valid.size()<<endl;
    cout<<"\n\tTest Data Size      \t"<<X_test.size()<<"x"<<X_test[0].size()<<endl;
    cout<<"\tTest Labels Size    \t"<<Y_test.size()<<endl;
    
    //defining architecture hyperparameters
    //vector<int> hidden_sizes = {128,64,32};
    vector<int> hidden_sizes = {128,32};
    //vector<int> hidden_sizes = {64,64,64,64};
    int input_size = X_train[0].size();
    int output_size = 10;
    string hidden_activation = "tanh";
    string output_activation = "softmax";
    
    //Creating the Model and initializing the weights.
    Network model(hidden_sizes,input_size,output_size,hidden_activation,output_activation);
    model.printArchitecture();
    cout<<"\nInitializing Model Parameters with Xavier Initialization...\n\n";
    model.InitializeWeights();

    //Defining the training hyperparmeters and training the model.
    Optimiser optim;
    double lr = 1e-3;
    int epochs = 10;
    double weight_decay = 0.5;
    int batch_size = 64;
    optim.rmsprop(X_train,Y_train,X_valid,Y_valid,model,lr,epochs,batch_size,weight_decay);
 
    //Evalute the model over test data.
    bool print_stats = true;
    int test_mode = 1;
    cout<<endl;
    optim.evaluateModel(model,X_test,Y_test,"Test",print_stats,test_mode);

    return 0;
}

/*
    Issues Addressed :

            Issue 1 : Each epoch is taking ~30-40min. 
                
                => fixed : There was a line auto Ai = data; in forward pass, causing entire data to be
                            copied for 60ktimes per epoch.

                => Still took 180 sec, did some Ninja and compiler optimization :)

            Model Accuracy Not increasing :

                =>Fixed : The data was corrupt.



    Good to Haves:

            1. Dropout
            2. BN
*/

/*

avoid push back in forward, instead try use index
pytorch for MNIST, see time per epoch
try with fashion mnist in cpp
add openMP pragma
try to optimize the existing code further.

*/