## Script

#### S1:

Hello everyone, in today’s presentation, I am going to introduce my research topic, exploring the viability of the widespread adoption of RISC-V architecture in mobile phones.

#### S2:

So, what is RISC-V? RISC-V is an instruction set architecture published by UC-Bakery in 2010. As shown, instruction set architecture, or ISA is the bridge connecting software and hardware that defines what a processor can do. You may have been very familiar with ARM and x86. And they are the current dominant ISA in the phones and PC areas. Today, it’s difficult to find a mobile phone without ARM processors inside. 

However, RISC-V has some properties that make it potentially outperform ARM and x86. And my research goal is basically to determine whether RISC-V can become strong enough to fight against ARM and enter the mobile phone area, so that in the future you may have your phone based on RISC-V architecture too.

#### S3:

RISC-V has three properties that the current ARM and x86 don’t have. First of all, RISC-V is open-source under the BSD License [5], which means everyone can use and modify it without paying a fee. Currently, if you want to make an ARM chip, you have to pay the Arm company a high licence fee [1]. Due to its openness, everyone is allowed to participate in the development of RISC-V, such as fixing security bugs or creating new functionality, making it more dynamic than ARM and x86 [6].

Secondly, RISC-V has a very cool modular design, which gives you high flexibility in building your chips. RISC-V defines four basic instruction sets and various official and custom instruction extensions to provide more functionality [2]. Every RISC-V processor must consist of one of the four basic instruction sets but can have none or multiple instruction-set extensions [2] (Point out the graph), just like building blocks. This allows you only to contain the instruction you need in your chip to suit your specific needs. In contrast, ARM and x86 processors usually have the one-size-fits-all program that provides as much as functionally in their chip to suit various application scenarios, but individual application scenarios may not need all the functions, which is wasting money and energy consumption [6]. Due to this, RISC-V has been widely implemented in embedded or domain-specific processors that are sensitive to cost and power. However, RISC-V is still in its infancy in other areas, such as high-performance and general-purpose processors that are usually used in PC, mobile phones, and servers [2].

#### S4:

The third property is simplicity. The basic instruction set of RISC-V only contains 47 instructions [8]. Compared to ARM and x86, which have been developed and iterated for decades, RISC-V hardly needs to consider any backward compatibility, such as retaining old instructions to support outdated features [1], as it is still relatively new. This is why RISC-V is so lightweight. For example, the official ARMv8-A specification document has 8528 pages for the first volume alone. In contrast, The RISC-V specification document consists of only two volumes with a total of 329 pages [1], which lower the barrier for people to learn RISC-V and participate in its development.

#### S5:

However, having some attractive properties is not enough for RISC-V to succeed in the mobile phone area. Therefore, my research will cover the hardware ecosystem, software ecosystem, security, and other external factors of RISC-V to find out the advantages and challenges for RISC-V architecture to be adopted in mobile phones.

Specifically, in the hardware ecosystem part, I am exploring the performance of Leading RISC-V Processors, the availability of RISC-V Processors for Mobile Phones, and academic and Industrial Activeness in RISC-V Extensions Research. For example, Xuantie-910 is a currently leading RISC-V processor that is capable of running Android. I can then do a performance analysis or find out-of-the-shelf analysis from the literature to compare it will the current ARM processors for phones. The reason for researching performance is that the processors' performance is one of the critical factors for customers to choose a phone.

I will also explore the software ecosystem of RISC-V, which includes Emulators, OSs, compilers, and development toolkit supports and the availability and industrial willingness to develop RISC-V-based Application software. Having a good software ecosystem is essential for the success of an ISA. For example, in recent years, relying on its robust software ecosystem, Apple has enabled its ARM-based PCs to challenge traditional x86 PCs.

Other external factors are also worth discovering, such as the competition between RISC-V and Arm. For example, in 2019, Arm took an anti-strategy that made the instruction set for its embedded processors Cortex-M customisable. This may weaken the advantages of RISC-V because customizability is one of the RISC-V “selling points”.

#### S6:

In order to research these sub-areas, I have done an extensive literature review to find out already known advantages and challenges for RISC-V mobile phones from the perspective of the software ecosystem, hardware ecosystem, security, and external challenges. The literature has revealed some research gaps that need to be filled before answering my research question fully and rigorously. Therefore, I will also look for news articles, industry reports, and summits to gather more up-to-date and detailed information. Finally, primary research will be conducted to fill the remaining gap, such as surveying the willingness of app developers to adapt RISC-V and benchmarking the leading general-purpose RISC-V and ARM processor.


#### S7:

My literature review was conducted by visiting the digital database of ACM, CNKI, and IEEE. Throughout the literature, some articles have been found to discuss the prospect of RISC-V explicitly. However, most are from a general perspective or the view of Internet of Things (IoT) applications instead of focusing on mobile phones.

By 18 May 2023, searching with the combination of the keywords “RISC-V” and “mobile phone” on ACM, CNKI, and IEEE would find no result, even though you can still find many personal videos and blog articles online explicitly discussing the RISC-V mobile phone prospect. Therefore, the main research goal is to academically relate RISC-V and mobile phones. 

#### S8:

Although the literature may not be directly relevant to mobile phones, some of them still provide me with valuable evidence to answer my research question.

Mezger (me zi ge) et al.’s (at-tau) article, “A Survey of the RISC-V Architecture Software Support “, provides the background knowledge of both hardware features and the software ecosystem of RISC-V to my research. The article first surveys RISC-V's current research progress and implementation of RISC-V hardware, including processor core, Socs, and emulators. Then it analyses the software ecosystem, including OS and development tools. Finally, it demonstrates how RISC-V hardware's security, reliability, and low-power feature promotes a good software ecosystem.

Liu et al.’s article provide a side-by-side benchmark comparison between ARM, RISC-V, and MIPS processors. This is closely relevant to my research because people do care about performance when buying phones. However, there is a limitation in this paper where the benchmark is done through emulation instead of running on physical chips. This is one of the research gaps revealed from the literature review.

#### S9:

The outcome of this research holds significance for researchers, investors, and engineers because they may want to know whether RISC-V mobile phone will potentially succeed before investing a lot of money and work in its development.

Adopting RISC-V in mobiles requires considerable funding support, especially for developing RISC-V chips. However, investors may have limited time or domain knowledge to delve into the topic.

For engineers, the research outcome can highlight the strengths and weaknesses of RISC-V in mobile phones, which may provide board guidance when making their RISC-V software or hardware product to maximise advantages.

Adopting RISC-V architecture into mobile phones requires a lot of domain-specific research, such as how to make the Android kernel support RISC-V processors. The outcome of my research gives an overall view of RISC-V on mobile phones, which is a good starting point for those who want to research deeper. 

#### S10:

This is all about my presentation today. Thank you for listening and I hope you can support my research.


