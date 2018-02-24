function HOSVD1pass(X,n,s,k,retrieve)
    Xmats=unfold(X)
    G1,Q1=pass_1_sketch(X,Xmats,n,s,k,retrieve)
    Xhat1=recon_Xhat(G1,Q1)
    return Xhat1
end

function HOSVD2pass(X,n,k,retrieve)
    Xmats=unfold(X)
    G,Q=pass_2_sketch(X,Xmats,n,k,retrieve)
    Xhat=recon_Xhat(G,Q)
    return Xhat
end

function run1passIteration(IterationsPerDim,lo,hi,numdim)
    dim = linspace(lo,hi,numdim)
    intdim=[]
    for dd in dim
        push!(intdim,trunc(Int,dd))
    end

    error1=[]
    time1=[]

    for i in dim
        error_1=[]
        time_1=[]
        r=trunc(Int,i/100+1) ######CUSTOMIZE RANK#####
        rank=(r,r,r)
        retrieve=rank #CUSTOMIZE RETRIEVE
        n=(trunc(Int,i),trunc(Int,i),trunc(Int,i))
        k=(2*r,2*r,2*r)
        s=(4*r,4*r,4*r)
        for iteration in 1:IterationsPerDim
            X=newlowranktensor(n,rank)
            Xhat1,time,byte,gc,other= @timed HOSVD1pass(X,n,s,k,retrieve)
            push!(error_1,vecnorm(Xhat1-X)^2/vecnorm(X)^2) #push 1pass error into error_1
            push!(time_1,time)
        end
        push!(error1,mean(error_1)) #push the average of all 10 iterations with dim i into error1 for 1pass
        push!(time1,mean(time_1))
    end
    return error1,time1
end

function run2passIteration(IterationsPerDim,lo,hi,numdim)
    dim = linspace(lo,hi,numdim)
    intdim=[]
    for dd in dim
        push!(intdim,trunc(Int,dd))
    end

    error2=[]
    time2=[]

    for i in dim
        error_2=[]
        time_2=[]
        r=trunc(Int,i/100+1) #####CUSTOMIZE RANK#######
        rank=(r,r,r)
        retrieve=rank ##CUSTOMIZE RETRIEVE
        n=(trunc(Int,i),trunc(Int,i),trunc(Int,i))
        k=(2*r,2*r,2*r)
        for iteration in 1:IterationsPerDim
            X=newlowranktensor(n,rank)
            Xhat,time,byte,gc,other= @timed HOSVD2pass(X,n,k,retrieve)
            push!(error_2,vecnorm(Xhat-X)^2/vecnorm(X)^2) #push 2pass error into error_2
            push!(time_2,time)
        end
        push!(error2,mean(error_2)) #push the average of all 10 iterations with dim i into error1 for 1pass
        push!(time2,mean(time_2))
    end
    return error2,time2
end

function runhosvd(lo,hi,numdim)
    dim = linspace(lo,hi,numdim)
    intdim=[]
    for dd in dim
        push!(intdim,trunc(Int,dd))
    end

    errorhosvd=[]
    timehosvd=[]
    for i in dim
        error_2=[]
        time_2=[]
        r=trunc(Int,i/100+1) #####CUSTOMIZE HOW TO CHOOSE RANK######
        rank=(r,r,r)
        retrieve=rank #####CUSTOMIZE RETRIEVAL
        #Calculate error from HOSVD package
        X=newlowranktensor(n,rank)
        Xh,time,byte,gc,other = @timed hosvd(X,reqrank=retrieve) #we only use the calculated Xh value and Time value
            core=Xh.cten
            feature=Xh.fmat
            Xhrecon=recon_Xhat(core,feature)
        push!(errorhosvd,vecnorm(Xhrecon-X)^2/vecnorm(X)^2) #push dim i into errorhosvd for HOSVD
        push!(timehosvd,time)
    end
    return errorhosvd,timehosvd
end
