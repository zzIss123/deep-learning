1. 配置静态ip并连接xshell

启动慢：(1) vi /etc/ssh/sshd_config

​	UseDNS yes   --->   UseDNS no

(2) systemctl restart sshd

(3)<img src="X:\机器学习\program\my_program\note\image-20230329133804472.png" alt="image-20230329133804472" style="zoom:70%;" />

<img src="X:\机器学习\program\my_program\note\image-20230329133852524.png" alt="image-20230329133852524" style="zoom:70%;" />

<img src="X:\机器学习\program\my_program\note\image-20230329134129664.png" alt="image-20230329134129664" style="zoom:67%;" />

(4) cd /etc/sysconfig/network-scripts/

![image-20230329134546065](X:\机器学习\program\my_program\note\image-20230329134546065.png)



vi ifcfg-eno16777736

![image-20230329134943743](X:\机器学习\program\my_program\note\image-20230329134943743.png)



reboot

2. 配置公网

<img src="X:\机器学习\program\my_program\note\image-20230329135348562.png" alt="image-20230329135348562" style="zoom:50%;" />

<img src="X:\机器学习\program\my_program\note\image-20230329135416508.png" alt="image-20230329135416508" style="zoom:50%;" />



<img src="X:\机器学习\program\my_program\note\image-20230329135622232.png" alt="image-20230329135622232" style="zoom:80%;" />

cd /etc/sysconfig/network-scripts/

vi ifcfg-eno16777736

reboot

![image-20230329135753476](X:\机器学习\program\my_program\note\image-20230329135753476.png)

![image-20230329140502690](X:\机器学习\program\my_program\note\image-20230329140502690.png)

