package org.pytorch.serve;
import org.pytorch.serve.job.Job;
import java.util.concurrent.LinkedBlockingDeque;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.TimeUnit;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.pytorch.serve.util.messages.InputParameter;

public class Scheduler {
    private Logger logger = LoggerFactory.getLogger(ModelServer.class);

    private int counter;
    
    private int gpuNumber;
    private int modelNumber;

    // store the jobs for later scheduling
    private CopyOnWriteArrayList<Job> jobList;
    // deque for the scheduled job
    private LinkedBlockingDeque<Job> jobDeque;

    class SchedulerJob {
        private Logger logger = LoggerFactory.getLogger(SchedulerJob.class);
    
        private String modelName;
        private String jobID;
        private Integer executedId;
    
        public SchedulerJob(Integer eid, String model_name, String jid) {
            modelName = model_name;
            jobID = jid;
            executedId = eid;
    
        }
    }

    public Scheduler() {
        logger.info("XXXXXXXXX start the scheduler");
        jobDeque = new LinkedBlockingDeque<Job>(100);
        jobList = new CopyOnWriteArrayList<Job>();
        counter = 0;
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

    public void schedule() {
        Job target = jobList.get(jobList.size() - 1);
        jobDeque.offer(target);
    }

    public boolean addJob(Job job) {
        logger.info("XXXXXXXXX add new job in scheduler {}, {}, {}",
            counter++, job.getModelName(), job.getJobId());
        jobList.add(job);
        schedule();

        return true;
    }

    public void addFirst(Job job) {
        logger.info("XXXXXXXXX add first new job in scheduler {}, {}, {}",
            counter++, job.getModelName(), job.getJobId());
        jobList.add(0, job);
    }
}