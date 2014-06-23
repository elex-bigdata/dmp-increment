package com.elex.dmp.utils;
import java.io.BufferedInputStream;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IOUtils;


public class HdfsUtils {

	public static void upFile(FileSystem fs,Configuration conf,String localFile,String hdfsPath) throws IOException{
		InputStream in=new BufferedInputStream(new FileInputStream(localFile));
		OutputStream out=fs.create(new Path(hdfsPath));
		IOUtils.copyBytes(in, out, conf);
		out.close();
		in.close();

	}

	public static void downFile(FileSystem fs,Configuration conf,String hdfsPath, String localPath) throws IOException{
		InputStream in=fs.open(new Path(hdfsPath));
		OutputStream out=new FileOutputStream(localPath);
		IOUtils.copyBytes(in, out, conf);
		out.close();
		in.close();
		
	}
	
	public static void makeDir(FileSystem fs,String hdfsPath) throws IOException{
		Path path = new Path(hdfsPath);
		if (fs.exists(path)) {
	        fs.delete(path, true);
	      }
		fs.mkdirs(path);
	}

	public static void delFile(FileSystem fs,String hdfsPath) throws IOException{
		fs.delete(new Path(hdfsPath), true);
	}
	
	public static void backupFile(FileSystem fs,Configuration conf,String src,String dst) throws IOException{
		Path srcP = new Path(src);
		Path dstP = new Path(dst);
		org.apache.hadoop.fs.FileUtil.copy(fs, srcP, fs, dstP, true, conf);
		
	}
	
}