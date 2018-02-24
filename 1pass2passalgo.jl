Pkg.add("TensorToolbox");
Pkg.add("TensorOperations");
Pkg.add("TensorDecompositions");
Pkg.add("PyPlot");

export newlowranktensor,pass_2_sketch,pass_1_sketch,unfold,recon_Xhat

using TensorOperations
#Generate random low-rank tensor with specified Tucker Rank and dimensions n[]
function newlowranktensor(n::Tuple,rank::Tuple)
    d = length(n); #order K tensor
    origG=randn(rank)
    X=origG
    for dd in 1:d
        X=ttm(X,randn(n[dd],rank[dd]),dd)
    end
    return X
end

#Returns the unfolding of tensor X
using TensorToolbox
function unfold(X)
    Xmats = Matrix[]
    for dd in 1:length(size(X))
        push!(Xmats, tenmat(X,dd))
    end
    return Xmats
end

#Perform 2 pass sketch (on the Matricization of X):
function pass_2_sketch(X,Xmats,n,k,retrieve)
#Xmats=unfolding of X; n=tuple of dimensions of X; rank is desired rank array, k is array with integers such that rank[i]<=k[i]<n[i].
    W=Matrix[]
    d=length(n)
    for dd in 1:d #for each unfolding of X
        n_W=[]
        for ee in 1:d
            if ee!=dd
                append!(n_W,n[ee]) #create n_W, array with dimensions of W
            end
        end

        psi=randn(prod(n_W), k[dd]) #create random sketch with size k
        Xmatr=Xmats[dd]
        W_r=Xmatr*psi #apply sketch matrix psi to get another entry in W
        push!(W, W_r)
    end
    G = X
    Q=Matrix[]
    for i in 1:(size(W)[1])
        Q_i,R= qr(W[i])
        Q_i=Q_i[:,1:retrieve[i]]
        G = ttm(G,transpose(Q_i),i) #Update G: G= G*Q[i] in the ith order; only use r vectors in Q to obtain r-rank factorization
        push!(Q,Q_i)
    end
    return G,Q
end

#Return the reconstruction of Xhat from G and arms Q
function recon_Xhat(G,Q)
    Xhat = G
    for dd in 1:length(size(G))
        Xhat = ttm(Xhat,Q[dd],dd)
    end
    return Xhat
end

function pass_1_sketch(X,Xmats,n,s,k,retrieve)
    W=Matrix[]
    Q=Matrix[]
    Y=Matrix[]
    Z=X
    d=length(n)
    psi=[randn(s[dd],n[dd]) for dd=1:d]
    omega=[randn(k[dd],trunc(Int,prod(n)/n[dd])) for dd=1:d]
    for dd in 1:d #for each unfolding of X
        push!(W,transpose(omega[dd]*transpose(Xmats[dd])))  #sketch the arms (W)
        q_dd,R=qr(W[dd])
        push!(Q,q_dd[:,1:retrieve[dd]]) #(Q)
        push!(Y,pinv(psi[dd]*(Q[dd]))) #contract Q's with Psi's
        Z=ttm(Z,psi[dd],dd) #sketch the core (Z)
    end
    G=Z
    for dd in 1:d
        G=ttm(G,Y[dd],dd)
    end
    return G,Q
end
