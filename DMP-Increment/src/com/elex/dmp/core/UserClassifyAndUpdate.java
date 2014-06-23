package com.elex.dmp.core;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.hbase.HBaseConfiguration;
import org.apache.hadoop.hbase.KeyValue;
import org.apache.hadoop.hbase.client.Get;
import org.apache.hadoop.hbase.client.HTable;
import org.apache.hadoop.hbase.client.Put;
import org.apache.hadoop.hbase.client.Result;
import org.apache.hadoop.hbase.io.ImmutableBytesWritable;
import org.apache.hadoop.hbase.mapreduce.TableMapReduceUtil;
import org.apache.hadoop.hbase.util.Bytes;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.SequenceFile.Reader;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.util.Tool;
import org.apache.mahout.common.iterator.sequencefile.PathFilters;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.SparseRowMatrix;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.Vector.Element;
import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.math.function.Functions;
import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import com.elex.dmp.utils.PropertiesUtil;

public class UserClassifyAndUpdate extends Configured implements Tool {
		
	/**
	 * @param args
	 */
	public static void main(String[] args) {
		// TODO Auto-generated method stub

	}

	@Override
	public int run(String[] args) throws Exception {
		Configuration conf = HBaseConfiguration.create();
		conf.set("model", args[0]);
		conf.set("dict", args[1]);
		conf.set("dump", args[2]);
		Job job = Job.getInstance(conf, "UserClassifyAndUpdate");
		job.setJarByClass(UserClassifyAndUpdate.class);
		job.setMapperClass(UserClassifyMapper.class);
		job.setMapOutputKeyClass(ImmutableBytesWritable.class);
		job.setMapOutputValueClass(Put.class);
		job.setInputFormatClass(SequenceFileInputFormat.class);		
		FileInputFormat.addInputPath(job, new Path(args[3]));
		TableMapReduceUtil.initTableReducerJob("dmp_user_classify", null, job);
		return job.waitForCompletion(true) ? 0 : 1;
	}

	static class UserClassifyMapper extends Mapper<Text, VectorWritable, ImmutableBytesWritable, Put> {
		
		private TopicModel model;
		private Map<Integer,Integer> dmpList = new HashMap<Integer,Integer>();
		private HTable uc;
		private List<Integer> ac;
		private List<String> dt;
		private int newAc=0,size=0;
		private double weightPrivious,weightNow;
		
		@Override
		protected void setup(Context context) throws IOException,InterruptedException {
			Configuration conf = context.getConfiguration();
			FileSystem fs = FileSystem.get(conf);
			
			//获取字典长度，由于通过map方法中value.get.size()为0.
			SequenceFile.Reader reader=new SequenceFile.Reader(conf,Reader.file(new Path(conf.get("dict"))));
			while(reader.next(new Text())){
				size++;
			}
			reader.close();
			
			//初始化模型文件			
			FileStatus[] statuses = fs.listStatus(new Path(conf.get("model")),PathFilters.partFilter());//
			Path[] modelPaths = new Path[statuses.length];
			for (int i = 0; i < statuses.length; i++) {
				modelPaths[i] = statuses[i].getPath();
			}
			model = new TopicModel(conf, 0.0001, 0.0001,new String[] { conf.get("dict") }, 3, 1.0f,modelPaths);
			
			//初始化dmp-dsp映射表
			BufferedReader br=new BufferedReader(new InputStreamReader(fs.open(new Path(conf.get("dump")))));
			String line = null;
			int i = 0;
			while (i < Integer.parseInt(PropertiesUtil.getNumTopics()) && (line = br.readLine()) != null) {
				try {
					JSONObject jsonObject = new JSONObject(line);
					JSONArray dspList = jsonObject.getJSONArray("dspMap");
					
					dmpList.put(jsonObject.getInt("id"),Integer.parseInt(dspList.getJSONObject(0).keys().next().toString()));
				} catch (JSONException e) {
					e.printStackTrace();
				}

				i++;
			}
			br.close();
			

			conf = HBaseConfiguration.create(conf);
			uc = new HTable(conf, "dmp_user_classify");
			uc.setAutoFlush(false);
			
			ac = new ArrayList<Integer>();
			dt = new ArrayList<String>();
		}
		
		@Override
		public void map(Text uid, VectorWritable value, Context context)throws IOException {
			int numTopics = Integer.parseInt(PropertiesUtil.getNumTopics());			
		    
			//用户分类
			Vector docTopics = new DenseVector(new double[numTopics]).assign(1.0 /numTopics);
		    Matrix docModel = new SparseRowMatrix(numTopics, size);
		    int maxIters = Integer.parseInt(PropertiesUtil.getClusterMaxIter());		    
		    for (int i = 0; i < maxIters; i++) {
		      model.trainDocTopicModel(value.get(), docTopics, docModel);
		    }
		    
		    
		    //获取已有的用户分类信息
		    ac.clear();
		    dt.clear();
		    Get get = new Get(Bytes.toBytes(uid.toString()));
		    get.setMaxVersions(2);
		    get.addFamily(Bytes.toBytes("uc"));
			Result r = uc.get(get);
			if(!r.isEmpty()){
				for (KeyValue kv : r.raw()) {
					if ("uc".equals(Bytes.toString(kv.getFamily())) && "AC".equals(Bytes.toString(kv.getQualifier()))) {
						ac.add(Integer.parseInt(Bytes.toString(kv.getValue())));
					}
					if ("uc".equals(Bytes.toString(kv.getFamily())) && "dt".equals(Bytes.toString(kv.getQualifier()))) {	
						dt.add(Bytes.toString(kv.getValue()));
					}	
				}
			}
			
			
			//更新用户分类信息			
			if(ac.size()==2 && dt.size()>0){
				Vector previous = new DenseVector(numTopics);
				for(String kv:dt.get(0).split(",")){
					String[] w = kv.split(":");
					previous.setQuick(Integer.parseInt(w[0]), Double.parseDouble(w[1]));				
				}
				
				newAc = ac.get(0)+ac.get(1);
				weightPrivious = (double)ac.get(1)/(double)newAc;
				weightNow = (double)ac.get(0)/(double)newAc;
				
				previous.assign(Functions.MULT, weightPrivious);
				docTopics.assign(Functions.MULT, weightNow);
				docTopics.assign(previous, Functions.PLUS);
			}else if(ac.size()==1){
				newAc = ac.get(0);
			}
			
			
			//入库
			Put put = new Put(Bytes.toBytes(uid.toString()));
			put.add(Bytes.toBytes("uc"), Bytes.toBytes("AC"), Bytes.toBytes(Integer.toString(newAc)));
			put.add(Bytes.toBytes("uc"), Bytes.toBytes("M"), Bytes.toBytes(Integer.toString(dmpList.get(docTopics.maxValueIndex()))));
			put.add(Bytes.toBytes("uc"), Bytes.toBytes("dt"), Bytes.toBytes(getStrFromVector(docTopics)));
					
		    try {
				context.write(new ImmutableBytesWritable(Bytes.toBytes(uid.toString())), put);
			} catch (InterruptedException e) {
				e.printStackTrace();
			}
			
		}
		
		@Override
		protected void cleanup(Context context) throws IOException,
				InterruptedException {
			uc.close();
		}
		
		public String getStrFromVector(Vector v){
			DecimalFormat df = new DecimalFormat("0.000000");
			StringBuffer str = new StringBuffer();
			Iterator<Element> it;
			Element e;
			it = v.iterator();
			str.delete(0, str.length());

			while (it.hasNext()) {
				e = it.next();
				str.append(e.index()).append(":").append(df.format(e.get())).append(",");				
			}
			return str.toString().substring(0, str.toString().length()-1);
		}
										
	}
	
}
