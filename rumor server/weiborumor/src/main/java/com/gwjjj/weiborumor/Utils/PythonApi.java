package com.gwjjj.weiborumor.Utils;

import java.io.*;
import java.net.InetAddress;
import java.net.Socket;
import java.util.ArrayList;
import java.util.List;

public class PythonApi {

    public static String doPython(List<String> list){
        Socket socket = null;
        try {
            InetAddress addr = InetAddress.getLocalHost();
            String host=addr.getHostName();
            //String ip=addr.getHostAddress().toString(); //获取本机ip
            //log.info("调用远程接口:host=>"+ip+",port=>"+12345);
            // 初始化套接字，设置访问服务的主机和进程端口号，HOST是访问python进程的主机名称，可以是IP地址或者域名，PORT是python进程绑定的端口号
            socket = new Socket(host,12345);
            // 获取输出流对象
            OutputStream os = socket.getOutputStream();
            PrintStream out = new PrintStream(os);
            // 发送内容
            for (int i = 0; i< list.size();i++) {
                String s = list.get(i);
                out.print(s + ",");
            }
            // 告诉服务进程，内容发送完毕，可以开始处理
            out.print("over");

            // 获取服务进程的输入流
            InputStream is = socket.getInputStream();
            BufferedReader br = new BufferedReader(new InputStreamReader(is,"utf-8"));
            String tmp = null;
            StringBuilder sb = new StringBuilder();
            // 读取内容
            while((tmp=br.readLine())!=null)
                sb.append(tmp).append('\n');
            System.out.print(sb);
            // 解析结果
            String re = sb.toString();
            String resultStr = re.substring(2,4) + "." + re.substring(5,7);
            if(resultStr.charAt(0) == '0'){
                return resultStr.substring(1);
            }
            return resultStr;
            //JSONArray res = JSON.parseArray(sb.toString());
        } catch (IOException e) {
            e.printStackTrace();
        }  finally {
            try {
                if(socket!=null) socket.close();
            } catch (IOException e) {
                e.printStackTrace();
            }
            System.out.println("远程接口调用结束");
        }
        return "";
    }

    public static void main(String[] args) {
        ArrayList<String> strs = new ArrayList<>();
        strs.add("得到答复我");
        strs.add("暗室逢灯微风巍峨我发的");
        strs.add("士大夫网咯德拉维拉斯");
        strs.add("都无法vv微风威锋网");
        String s = PythonApi.doPython(strs);
        System.out.println(s);
    }
}
