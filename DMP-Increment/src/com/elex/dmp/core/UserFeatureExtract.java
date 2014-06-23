package com.elex.dmp.core;

import java.io.IOException;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.hbase.HBaseConfiguration;
import org.apache.hadoop.hbase.KeyValue;
import org.apache.hadoop.hbase.client.Get;
import org.apache.hadoop.hbase.client.HTable;
import org.apache.hadoop.hbase.client.Put;
import org.apache.hadoop.hbase.client.Result;
import org.apache.hadoop.hbase.client.Scan;
import org.apache.hadoop.hbase.io.ImmutableBytesWritable;
import org.apache.hadoop.hbase.mapreduce.TableMapReduceUtil;
import org.apache.hadoop.hbase.mapreduce.TableMapper;
import org.apache.hadoop.hbase.util.Bytes;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.hadoop.util.GenericOptionsParser;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;

import com.elex.dmp.utils.PropertiesUtil;


public class UserFeatureExtract extends Configured implements Tool {

	public static class MyMapper extends TableMapper<Text, Text> {
		
		private Text uid = new Text();
		

		@Override
		protected void map(ImmutableBytesWritable key, Result values,Context context) throws IOException, InterruptedException {
			uid.set(getUid(key));
			for (KeyValue kv : values.raw()) {
				if ("ua".equals(Bytes.toString(kv.getFamily())) && "url".equals(Bytes.toString(kv.getQualifier()))) {
					
					context.write(uid, new Text(kv.getValue()));
				}
			}

			
		}

		private byte[] getUid(ImmutableBytesWritable key) {

			return Bytes.tail(key.get(), key.get().length-9);
		}

	}

	
	
	 /*
	  * Reducer类不能加final标记，否则会报错No such method 。。。。。init()
	  */
	public static class UserFeatureReducer extends Reducer<Text, Text, Text, Text> {

				

		private HTable ud,uc;
		private Configuration configuration;
		//private StringBuilder keywords;
		private String Lang="";
		private int KWNum;
		private String words;
		private int actionCount;
		
		@Override
		protected void setup(Context context) throws IOException,InterruptedException {
			configuration = HBaseConfiguration.create();
			ud = new HTable(configuration, "dmp_url_detail");
			uc = new HTable(configuration, "dmp_user_classify");
			uc.setAutoFlush(false);
			//keywords = new StringBuilder();
		}

		@Override
		protected void reduce(Text uid, Iterable<Text> urlList,Context context) throws IOException, InterruptedException {
			Put put = new Put(Bytes.toBytes(uid.toString()));//new Put(uid.getBytes())会导致写入hbase中的用户id和写入特征文件中的用户id不一致
			actionCount = 0;
			StringBuilder keywords = new StringBuilder();
			
			for(Text key:urlList){
				Get get = new Get(Bytes.add(Bytes.toBytes("T"),Bytes.toBytes(key.toString())));//注意！！！！Bytes.toBytes(key.toString())必须这么写,因为编码方式的差异
				get.addFamily(Bytes.toBytes("ud"));
				Result r = ud.get(get);
				if(!r.isEmpty()){
					for (KeyValue kv : r.raw()) {
						if ("ud".equals(Bytes.toString(kv.getFamily())) && "l".equals(Bytes.toString(kv.getQualifier()))) {							
							Lang = Bytes.toString(kv.getValue());							
						}
						if ("ud".equals(Bytes.toString(kv.getFamily())) && "k".equals(Bytes.toString(kv.getQualifier()))) {									
							KWNum = Bytes.toString(kv.getValue()).split(" ").length;
							words = Bytes.toString(kv.getValue());
						}	
					}
				}
				
				if(Lang.startsWith("en")){
					if(KWNum >= PropertiesUtil.getKeyWordNumber()){
						keywords.append(words+" ");						
						actionCount++;
						}
					}
			}
			
			put.add(Bytes.toBytes("uc"), Bytes.toBytes("AC"), Bytes.toBytes(Integer.toString(actionCount)));
			put.setWriteToWAL(false);
			uc.put(put);			
			context.write(uid, new Text(keywords.toString()));
		}
		
		@Override
		protected void cleanup(Context context) throws IOException,
				InterruptedException {
			ud.close();
			uc.close();//这一步尤其重要，因为设置了uc表不自动flush，如果数据量够小则需要通过close来触发flush
		}
	}
	
	

	/**
	 * @param args
	 * @throws Exception 
	 */
	public static void main(String[] args) throws Exception {
		 Configuration conf = new Configuration();
		 String[] otherArgs = new GenericOptionsParser(conf, args).getRemainingArgs();
		 int res = ToolRunner.run(new Configuration(), new UserFeatureExtract(), otherArgs);
		 System.exit(res);
	}

	@Override
	public int run(String[] args) throws Exception {
		Configuration conf = new Configuration();
        conf = HBaseConfiguration.create(conf);
        Job job = Job.getInstance(conf,"userFeatureExtract");
        job.setJarByClass(UserFeatureExtract.class);
        byte[] type =new byte[]{PropertiesUtil.getType()};
        long now =System.currentTimeMillis();
        long before = now - Long.parseLong(args[0])*60*1000;
        byte[] startRow = Bytes.add(type,Bytes.toBytes(before));
        byte[] stopRow = Bytes.add(type,Bytes.toBytes(now));       
		Scan s = new Scan();
		s.setStartRow(startRow);
		s.setStopRow(stopRow);
		s.setCaching(500);
		s.addColumn(Bytes.toBytes("ua"), Bytes.toBytes("url"));
		TableMapReduceUtil.initTableMapperJob("dmp_user_action", s, MyMapper.class,Text.class, Text.class, job);
		job.setMapOutputKeyClass(Text.class);
		job.setMapOutputValueClass(Text.class);
		job.setReducerClass(UserFeatureReducer.class);
		job.setOutputKeyClass(Text.class);
		job.setOutputValueClass(Text.class);
		FileOutputFormat.setOutputPath(job, new Path(args[1]));
		job.setOutputFormatClass(SequenceFileOutputFormat.class);//如果发现编译一直报错，可能是import了mapred包里的同名SequenceFileOutputFormat类。
		return job.waitForCompletion(true)?0:1;
	}

}
