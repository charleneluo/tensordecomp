function plotresults(lo,hi,numdim,IterationsPerDim)
    dim = linspace(lo,hi,numdim)
    errorhosvd,timehosvd=runhosvd(lo,hi,numdim)
    error2,time2=run2passIteration(IterationsPerDim,lo,hi,numdim)
    error1,time1=run1passIteration(IterationsPerDim,lo,hi,numdim)
    fig = figure("pyplot_subplot_column",figsize=(20,10))
    subplot(211) # Create first plot
    plot(dim, error1, color="red", linewidth=2.0, linestyle="--",label="1pass") #RED = 1 PASS
    plot(dim, error2, color="blue", linewidth=2.0,linestyle="--",label="2pass") #BLUE = 2 PASS
    plot(dim, errorhosvd, color="green",linewidth=2.0,linestyle="--",label="hosvd") #GREEN=HOSVD
    title("Error due to HOSVD from 1pass and 2pass Algorithms")
    ylabel("MSE")
    xlabel("Dimension")
    legend()
    subplot(212) # Create second plot
    ylabel("Seconds")
    xlabel("Dimension")
    plot(dim, time1, color="red", linewidth=2.0, linestyle="--",label="1pass") #RED = 1 PASS
    plot(dim, time2, color="blue", linewidth=2.0,linestyle="--",label="2pass") #BLUE = 2 PASS
    plot(dim, timehosvd, color="green",linewidth=2.0,linestyle="--",label="hosvd") #GREEN=HOSVD
    title("Runtime of HOSVD from 1pass and 2pass Algorithms")
    legend()
    fig[:canvas][:draw]() # Update the figure
    gcf() # Needed for IJulia to plot inline
    println("-------------------ERRORS---------------------")
    println("1pass")
    println(error1)
    println("2pass")
    println(error2)
    println("HOSVD package")
    println(errorhosvd)
    println("-------------------TIMES-------------------")
    println("1pass")
    println(time1)
    println("2pass")
    println(time2)
    println("HOSVD package")
    println(timehosvd)
end
