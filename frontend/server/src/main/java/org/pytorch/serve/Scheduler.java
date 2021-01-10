package org.pytorch.serve;
import org.pytorch.serve.job.Job;
import java.util.concurrent.LinkedBlockingDeque;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.TimeUnit;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.pytorch.serve.util.messages.InputParameter;
import org.pytorch.serve.SchedulerThread;
import java.util.ArrayList;
import java.util.Collection;

public class Scheduler {
    private Logger logger = LoggerFactory.getLogger(ModelServer.class);

    private int counter;
    
    private int gpuNumber;
    private int modelNumber;

    // store the jobs for later scheduling
    private CopyOnWriteArrayList<Job> jobList;
    // deque for the scheduled job
    private LinkedBlockingDeque<Job> jobDeque;

    public Scheduler() {
        logger.info("XXXXXXXXX start the scheduler");
        jobDeque = new LinkedBlockingDeque<Job>(100);
        jobList = new CopyOnWriteArrayList<Job>();
        counter = 0;

        logger.info("XXXXXXXXX create the scheduler thread");
        Runnable runnable = new SchedulerThread(this);
        Thread thread = new Thread(runnable);
        thread.start();
    }

    public int getGpuNumber() {
        return gpuNumber;
    }

    public int getModelNumber() {
        return modelNumber;
    }

    public void addGpuNumber(int n) {
        gpuNumber += n;
    }

    public void addModelNumber(int n) {
        modelNumber += n;
    }

    public Job pollScheduledJob() {
        try {
            return jobDeque.poll(Long.MAX_VALUE, TimeUnit.MILLISECONDS);
        } catch(Exception e) {
            return null;
        }
    }

    public void schedule(long deadline) {
        
        ArrayList<Integer>  jobs = new ArrayList<Integer>(20);
        for(int i = 0; i < jobList.size(); i ++) {
            Job job = jobList.get(i);
            if (job.getDeadline() < deadline) {
                jobs.add(i);
            }
        }
        jobs.sort(null);

        for(int i = 0; i < jobs.size(); i ++) {
            int jobId = jobs.get(i);
            Job target = jobList.get(jobId);

            String layers = "0,0,0,0,0,0,0,0,0,0";
            if(target.getGPULayers() != null){
                layers = target.getGPULayers();
            }

            target.getPayload().addParameter(
                new InputParameter("gpu_layers", layers));
            jobDeque.offer(target);
        }

        for(int i = jobs.size() -1; i >= 0 ; i--) {
            jobList.remove(i);
        }
    }

    public boolean addJob(Job job) {
        logger.info("XXXXXXXXX add new job in scheduler {}, {}, {}",
            counter++, job.getModelName(), job.getJobId());
        jobList.add(job);

        return true;
    }

    public void addFirst(Job job) {
        logger.info("XXXXXXXXX add first new job in scheduler {}, {}, {}",
            counter++, job.getModelName(), job.getJobId());
        jobList.add(0, job);
    }
}