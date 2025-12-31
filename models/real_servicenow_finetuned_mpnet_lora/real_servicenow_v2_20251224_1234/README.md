---
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- dense
- generated_from_trainer
- dataset_size:5000
- loss:CosineSimilarityLoss
base_model: sentence-transformers/all-mpnet-base-v2
widget:
- source_sentence: INC0021792 PGI-SO-2403000155, PGI-SO-2406000745, PGI-SO-2406000756
    With Part Number is already Release by CPC (Part Center), But cannot Process “Acknowledge
    Reception” Because Until Now Status In Order Line Status is “Dlv.Processing” ,
    Please help to deeply check because this case is so many appearing, Thanks Closed
    by Caller
  sentences:
  - INC0044329 Users have problem to issue order in Glics due to no line code for
    item K14XZ0198EMVB-T. When checked in SAP, there is production version created.
    But when check model interface to Glics, there is no item for K14XZ0198EMVB-T.
    Please refer attached file as for your reference. Closed by Caller
  - "INC0045244 Requester : lee.hansoo@kr.panasonic.com\r\n\r\nAn error occurred while\
    \ billing software in December, and  \"Acct Determin Error\" was displayed. Please\
    \ investigate the cause and resolve it. Please note , as per your confirmation,\
    \ I have changed the validity date to past for ZBFE condition type to the material\
    \ used at line item 20 and then carried out new pricing at Invoice level so that\
    \ ZBFE has disappeared from billing document and also Journal entry has posted\
    \ as expected. I have maintained the validity date for ZBFE back to normal as\
    \ it was available in the system.\r\n\r\n Kindly work with master data team to\
    \ delete ZBFE condition type for material used at Item 20, so that it will not\
    \ populate for new orders which are going to create in future."
  - 'INC0042954 Dear Team


    Error occurred while processing the EDI transaction


    Interface       internal-stocks

    Subsidiary      PAU

    API Name        pana-sdesk-ext-eapi

    Flow Direction  outbound

    Source System   PAU

    End System      ZOHO

    File Name       No File Name

    Storage Path    No Attachment

    Error Source    pana-sdesk-ext-eapi

    Transaction ID  55814550-a78c-11ef-8b0c-02b4130d4440

    Error Summary   500 EXPRESSION

    Error Details   EXPRESSION: "org.mule.runtime.api.exception.MuleRuntimeException
    - Exception was found trying to retrieve the contents of file /Zoho/Live/Outbound/status/remote/SPR_Status_20241121090803362047.csv
    org.mule.runtime.api.exception.MuleRuntimeException: Exception was found trying
    to retrieve the contents of file /Zoho/Live/Outbound/status/remote/SPR_Status_20241121090803362047.csv
    Caused by: SFTP error (SSH_FX_FAILURE): Failure. at org.apache.sshd.sftp.client.impl.AbstractSftpClient.throwStatusException(AbstractSftpClient.java:277)
    at org.apache.sshd.sftp.client.impl.AbstractSftpClient.checkHandleResponse(AbstractSftpClient.java:299)
    at org.apache.sshd.sftp.client.impl.AbstractSftpClient.checkHandle(AbstractSftpClient.java:290)
    at org.apache.sshd.sftp.client.impl.AbstractSftpClient.open(AbstractSftpClient.java:589)
    at org.apache.sshd.sftp.client.impl.SftpInputStreamAsync.(SftpInputStreamAsync.java:75)
    at org.apache.sshd.sftp.client.impl.AbstractSftpClient.read(AbstractSftpClient.java:1196)
    at org.apache.sshd.sftp.client.SftpClient.read(SftpClient.java:909) at org.apache.sshd.sftp.client.SftpClient.read(SftpClient.java:905)
    at org.mule.extension.sftp.internal.connection.SftpClient.getFileContent(SftpClient.java:386)
    at org.mule.extension.sftp.internal.connection.SftpFileSystemConnection.retrieveFileContent(SftpFileSystemConnection.java:117)
    at org.mule.extension.sftp.internal.operation.SftpInputStream$SftpFileInputStreamSupplier.getContentInputStream(SftpInputStream.java:111)
    at org.mule.extension.sftp.internal.operation.SftpInputStream$SftpFileInputStreamSupplier.getContentInputStream(SftpInputStream.java:92)
    at org.mule.extension.sftp.internal.operation.AbstractConnectedFileInputStreamSupplier.getContentInputStream(AbstractConnectedFileInputStreamSupplier.java:90)
    at org.mule.extension.sftp.internal.operation.AbstractFileInputStreamSupplier.get(AbstractFileInputStreamSupplier.java:68)
    at org.mule.extension.sftp.internal.operation.AbstractFileInputStreamSupplier.get(AbstractFileInputStreamSupplier.java:36)
    at org.mule.extension.sftp.internal.stream.LazyStreamSupplier.lambda$new$1(LazyStreamSupplier.java:29)
    at org.mule.extension.sftp.internal.stream.LazyStreamSupplier.get(LazyStreamSupplier.java:42)
    at org.mule.extension.sftp.internal.util.LazyInputStreamProxy.getDelegate(LazyInputStreamProxy.java:29)
    at org.mule.extension.sftp.internal.util.LazyInputStreamProxy.read(LazyInputStreamProxy.java:48)
    at org.apache.commons.io.input.ProxyInputStream.read(ProxyInputStream.java:205)
    at org.mule.runtime.core.internal.streaming.bytes.AbstractInputStreamBuffer.consumeStream(AbstractInputStreamBuffer.java:111)
    at com.mulesoft.mule.runtime.core.internal.streaming.bytes.FileStoreInputStreamBuffer.consumeForwardData(FileStoreInputStreamBuffer.java:242)
    at com.mulesoft.mule.runtime.core.internal.streaming.bytes.FileStoreInputStreamBuffer.consumeForwardData(FileStoreInputStreamBuffer.java:205)
    at com.mulesoft.mule.runtime.core.internal.streaming.bytes.FileStoreInputStreamBuffer.doGet(FileStoreInputStreamBuffer.java:128)
    at org.mule.runtime.core.internal.streaming.bytes.AbstractInputStreamBuffer.get(AbstractInputStreamBuffer.java:93)
    at org.mule.runtime.core.internal.streaming.bytes.BufferedCursorStream.assureDataInLocalBuffer(BufferedCursorStream.java:126)
    at org.mule.runtime.core.internal.streaming.bytes.BufferedCursorStream.doRead(BufferedCursorStream.java:101)
    at org.mule.runtime.core.internal.streaming.bytes.AbstractCursorStream.read(AbstractCursorStream.java:124)
    at org.mule.runtime.core.internal.streaming.bytes.BufferedCursorStream.read(BufferedCursorStream.java:26)
    at org.mule.runtime.core.internal.streaming.bytes.ManagedCursorStreamDecorator.read(ManagedCursorStreamDecorator.java:101)
    at org.mule.weave.v2.el.SeekableCursorStream.biggerOrEqualThan(MuleTypedValue.scala:317)
    at org.mule.weave.v2.el.SeekableCursorStream.inMemory$lzycompute(MuleTypedValue.scala:330)
    at org.mule.weave.v2.el.SeekableCursorStream.inMemory(MuleTypedValue.scala:326)
    at org.mule.weave.v2.el.SeekableCursorStream.inMemoryStream(MuleTypedValue.scala:338)
    at org.mule.weave.v2.module.reader.UTF8StreamSourceReader.inMemoryReader(SeekableStreamSourceReader.scala:220)
    at org.mule.weave.v2.module.reader.SourceReader.requireClose(SourceReader.scala:123)
    at org.mule.weave.v2.module.reader.SourceReader.requireClose$(SourceReader.scala:123)
    at org.mule.weave.v2.module.reader.UTF8StreamSourceReader.requireClose(SeekableStreamSourceReader.scala:138)
    at org.mule.weave.v2.module.reader.ResourceManager.registerCloseable(ResourceManager.scala:29)
    at org.mule.weave.v2.model.EvaluationContext.registerCloseable(EvaluationContext.scala:64)
    at org.mule.weave.v2.model.EvaluationContext.registerCloseable$(EvaluationContext.scala:63)
    at org.mule.weave.v2.interpreted.DefaultExecutionContext.registerCloseable(ExecutionContext.scala:364)
    at org.mule.weave.v2.module.reader.SourceReader$.apply(SourceReader.scala:135)
    at org.mule.weave.v2.module.core.csv.reader.CSVReader.createCSVRootValue(CSVReader.scala:41)
    at org.mule.weave.v2.module.core.csv.reader.CSVReader.doRead(CSVReader.scala:36)
    at org.mule.weave.v2.module.reader.Reader.read(Reader.scala:37) at org.mule.weave.v2.module.reader.Reader.read$(Reader.scala:35)
    at org.mule.weave.v2.module.core.csv.reader.CSVReader.read(CSVReader.scala:26)
    at org.mule.weave.v2.el.MuleTypedValue.value(MuleTypedValue.scala:145) at org.mule.weave.v2.model.values.wrappers.DelegateValue.evaluate(DelegateValue.scala:23)
    at org.mule.weave.v2.model.values.wrappers.DelegateValue.evaluate$(DelegateValue.scala:23)
    at org.mule.weave.v2.el.MuleTypedValue.evaluate(MuleTypedValue.scala:50) at org.mule.weave.v2.module.core.json.writer.JsonWriter.doWriteValue(JsonWriter.scala:189)
    at org.mule.weave.v2.module.writer.Writer.writeValue(Writer.scala:62) at org.mule.weave.v2.module.writer.Writer.writeValue$(Writer.scala:48)
    at org.mule.weave.v2.module.core.json.writer.JsonWriter.writeValue(JsonWriter.scala:36)
    at org.mule.weave.v2.module.writer.WriterWithAttributes.writeAttributesAndValue(WriterWithAttributes.scala:29)
    at org.mule.weave.v2.module.writer.WriterWithAttributes.writeAttributesAndValue$(WriterWithAttributes.scala:14)
    at org.mule.weave.v2.module.core.json.writer.JsonWriter.writeAttributesAndValue(JsonWriter.scala:36)
    at org.mule.weave.v2.module.core.json.writer.JsonWriter.writeObject(JsonWriter.scala:101)
    at org.mule.weave.v2.module.core.json.writer.JsonWriter.doWriteValue(JsonWriter.scala:190)
    at org.mule.weave.v2.module.writer.Writer.writeValue(Writer.scala:62) at org.mule.weave.v2.module.writer.Writer.writeValue$(Writer.scala:48)
    at org.mule.weave.v2.module.core.json.writer.JsonWriter.writeValue(JsonWriter.scala:36)
    at org.mule.weave.v2.module.writer.DeferredWriter.doWriteValue(DeferredWriter.scala:77)
    at org.mule.weave.v2.module.writer.Writer.writeValue(Writer.scala:62) at org.mule.weave.v2.module.writer.Writer.writeValue$(Writer.scala:48)
    at org.mule.weave.v2.module.writer.DeferredWriter.writeValue(DeferredWriter.scala:17)
    at org.mule.weave.v2.module.writer.WriterHelper$.writeValue(Writer.scala:162)
    at org.mule.weave.v2.module.writer.WriterHelper$.writeAndGetResult(Writer.scala:140)
    at org.mule.weave.v2.interpreted.InterpretedMappingExecutableWeave.writeWith(InterpreterMappingCompilerPhase.scala:256)
    at org.mule.weave.v2.el.WeaveExpressionLanguageSession.evaluateWithTimeout(WeaveExpressionLanguageSession.scala:313)
    at org.mule.weave.v2.el.WeaveExpressionLanguageSession.$anonfun$evaluate$3(WeaveExpressionLanguageSession.scala:122)
    at org.mule.weave.v2.el.WeaveExpressionLanguageSession.doEvaluate(WeaveExpressionLanguageSession.scala:270)
    at org.mule.weave.v2.el.WeaveExpressionLanguageSession.evaluate(WeaveExpressionLanguageSession.scala:121)
    at org.mule.runtime.core.internal.el.dataweave.DataWeaveExpressionLanguageAdaptor$1.evaluate(DataWeaveExpressionLanguageAdaptor.java:372)
    at org.mule.runtime.core.internal.el.DefaultExpressionManagerSession.evaluate(DefaultExpressionManagerSession.java:105)
    at com.mulesoft.mule.runtime.core.internal.processor.SetVariableTransformationTarget.process(SetVariableTransformationTarget.java:40)
    at com.mulesoft.mule.runtime.core.internal.processor.TransformMessageProcessor.process(TransformMessageProcessor.java:99)
    at org.mule.runtime.core.api.util.func.CheckedFunction.apply(CheckedFunction.java:25)
    at org.mule.runtime.core.api.rx.Exceptions.lambda$checkedFunction$2(Exceptions.java:85)
    at org.mule.runtime.core.internal.util.rx.Operators.lambda$nullSafeMap$0(Operators.java:47)
    at reactor.core.publisher.FluxHandleFuseable$HandleFuseableSubscriber.onNext(FluxHandleFuseable.java:176)
    at org.mule.runtime.core.privileged.processor.chain.AbstractMessageProcessorChain$2.onNext(AbstractMessageProcessorChain.java:628)
    at org.mule.runtime.core.privileged.processor.chain.AbstractMessageProcessorChain$2.onNext(AbstractMessageProcessorChain.java:623)
    at reactor.core.publisher.FluxHide$SuppressFuseableSubscriber.onNext(FluxHide.java:137)
    at reactor.core.publisher.FluxPeekFuseable$PeekFuseableSubscriber.onNext(FluxPeekFuseable.java:210)
    at reactor.core.publisher.FluxOnAssembly$OnAssemblySubscriber.onNext(FluxOnAssembly.java:539)
    at reactor.core.publisher.FluxSubscribeOnValue$ScheduledScalar.run(FluxSubscribeOnValue.java:180)
    at reactor.core.scheduler.SchedulerTask.call(SchedulerTask.java:68) at reactor.core.scheduler.SchedulerTask.call(SchedulerTask.java:28)
    at java.util.concurrent.FutureTask.run(FutureTask.java:266) at org.mule.service.scheduler.internal.AbstractRunnableFutureDecorator.doRun(AbstractRunnableFutureDecorator.java:180)
    at org.mule.service.scheduler.internal.RunnableFutureDecorator.run(RunnableFutureDecorator.java:55)
    at java.util.concurrent.ThreadPoolExecutor.runWorker(ThreadPoolExecutor.java:1149)
    at java.util.concurrent.ThreadPoolExecutor$Worker.run(ThreadPoolExecutor.java:624)
    at java.lang.Thread.run(Thread.java:750, while reading `payload` as CSV. Trace:
    at main (Unknown)" evaluating expression: "%dw 2.0 output application/json ---
    { "correlationId": correlationId, "domain": Mule::p(''api.domain.serviceDesk''),
    "transmissionDate": now(), "company": vars.varMetadata.company, "interface": vars.varMetadata.interface,
    "interfaceType": vars.varMetadata.interfaceType, "source": vars.varMetadata.source,
    "target": vars.varMetadata.target, "action": vars.varMetadata.action, "status":
    vars.varMetadata.status, "requestPayload" : vars.varMetadata.payload default payload,
    "payloadType": vars.varMetadata.payloadType, "sourcePath": vars.varMetadata.sourcePath,
    "backupPath": vars.varMetadata.backupPath, "destinationPath": vars.varMetadata.destinationPath,
    "api": Mule::p("api.name"), "apiLayer": Mule::p("api.layer.experience"), "reprocessedFlag":
    vars.varMetadata.reprocessedFlag default "N", "errorSource": vars.varMetadata.errorSource
    default "", "errorCode": vars.varMetadata.errorCode default "", "errorMessage":
    vars.varMetadata.errorMessage default "", "errorPayload": vars.varMetadata.errorPayload
    default "" }".

    Comments


    Note: This is an automated mail, please do not reply.


    Thanks and Regards,


    APAC Support, Panasonic The setup for the file poller in outbound servicedesk
    is it does not do overwrite, thus, the issue encountered. As a workaround, the
    previous file has been deleted. As for the resolution, overwrite is now enabled.'
- source_sentence: "INC0048803 we got one error message from eWork, please check how\
    \ ework interface to BRS, we need resolve it asap Hi Trung Duc Duong san,\r\n\
    Good evening!\r\n \r\nAs communicated earlier, the eInvoice document 'INV10863'\
    \ is already posted and the initial posting failure error might be that the posting\
    \ data is locked by user BRS_NET at SAP side which caused posting failure. \r\n\
    \ \r\nHence confirmed from your end, we are proceeding to close this incident\
    \ ticket 'INC0048803'.\r\n\r\nThanks,\r\nKannan B"
  sentences:
  - "INC0062510 Customer material info record to factory wrong.\r\nWe found that the\
    \ EDI sent to the factory customer material info record incorrect, but customer\
    \ material info record PO is correct. as checking PO no change customer material.\
    \ \r\n\r\n400043699\r\n400043700\r\n400043697\r\n400043698\r\n400043696 \r\n\r\
    \nPlease see the details in the attached file. TH Users has no longer access for\
    \ ME28 and ME29N. The Related roles has been removed in MBP500"
  - 'INC0053196 Dear Team


    Error occurred while processing the EDI transaction. Please find the details below
    and attached is the file associated to the transaction.


    Interface       employee-minimaster-ridm

    Subsidiary      PAPAMY

    API Name        pana-global-hriq-sapi

    Flow Direction  inbound

    Source System   successFactors

    End System      RIDM

    File Name       ridm-empminimaster_45bb0000-0067-11f0-b545-c229612ff882.json

    Storage Path    /PAPAMY/ridm/employee-mini-master/ridm-empminimaster_45bb0000-0067-11f0-b545-c229612ff882.json

    Error Source    RIDM

    Transaction ID  47ccc900-0067-11f0-9922-a61a832da79c

    Error Summary   400 BAD_REQUEST

    Error Details   Error response from RIDM API

    Comments


    Note: This is an automated mail, please do not reply.


    Thanks and Regards,


    APAC Support, Panasonic Invalid data'
  - 'INC0058695 Dear Team


    Error occurred while processing the EDI transaction


    Interface

    Subsidiary      No Subsidiary

    API Name        pana-pagitp-mgmt-eapi

    Flow Direction  Inbound

    Source System   PAGITP

    End System      SAP S4Hana

    File Name       No File Name

    Storage Path    No Attachment

    Error Source    Mulesoft

    Transaction ID  52147250-36a9-11f0-ae4d-fa5087e70212

    Error Summary   500 INVALID_CREDENTIALS

    Error Details   Could not establish SFTP connection with host: ''10.86.48.62''
    at port: ''22'' - Error during login to mule2INDS@10.86.48.62

    Comments


    Note: This is an automated mail, please do not reply.


    Thanks and Regards,


    APAC Support, Panasonic scheduler job run every 2 mins, the connection has re-established'
- source_sentence: "INC0065441 Hi Team, \r\nCan you please repush idoc 5521358\r\n\
    It is successful in SAP but still stuck in SFDC \r\nCan we please know why this\
    \ has not come back to sfdc\r\n\r\nThanks 1st file encountered system exception\r\
    \nReprocess of file is successful."
  sentences:
  - "INC0050051 Dear Team\n\nError occurred while processing the EDI transaction\n\
    \nInterface       PO-ACK\nSubsidiary      NA\nAPI Name        inds-global-if-mgmt-papi\n\
    Flow Direction  Inbound\nSource System   SEGlink\nEnd System      SAP\nFile Name\
    \       APRK14L20250210153517351\nStorage Path    No Attachment\nError Source\
    \    Mulesoft\nTransaction ID  3b1f97c2-e779-11ef-aaa4-de4a2ed45e96\nError Summary\
    \   500 CONNECTIVITY\nError Details   ***********443/api/v1/mm/acknowledgment'\
    \ failed: Remotely closed.\nComments\n\nNote: This is an automated mail, please\
    \ do not reply.\n\nThanks and Regards,\n\nAPAC Support, Panasonic 1. Check the\
    \ ACK file from backup folder in common storage\r\n2. Reput in SEGLink folder\r\
    \n3. Verify in MBP"
  - "INC0052715 Dear support team,\r\n\r\nWe have problem when sending EDI as below:\r\
    \nDO: 4100028640\r\nTcode: y0gsd_0410\r\n\r\nPls check attached image to details.\r\
    \n\r\nThank you. Avoid using special character in the container number"
  - 'INC0057427 Dear Team


    Error occurred while processing the EDI transaction


    Interface

    Subsidiary      No Subsidiary

    API Name        pana-pagitp-mgmt-eapi

    Flow Direction  Inbound

    Source System   PAGITP

    End System      SAP S4Hana

    File Name       No File Name

    Storage Path    No Attachment

    Error Source    Mulesoft

    Transaction ID  dacf7225-5a99-4a4f-a8b6-eab66f4c4674

    Error Summary   500 SOURCE_RESPONSE_SEND

    Error Details   Client connection was closed

    Comments


    Note: This is an automated mail, please do not reply.


    Thanks and Regards,


    APAC Support, Panasonic No action needed. YEMM_ORDRSP iDoc posted successfully
    to SAP'
- source_sentence: "INC0048576 PASS 4614@id.panasonic.com (Karya Pendingin) (Partner\
    \ Portal)\r\nIs Cannot request approval for warranty discounts over 1 year for\
    \ PGI-WO-2501006104, \r\nThis is the step For Check :\r\n1).Open Work Order and\
    \ Click Submit For Approval ;\r\n2).Click Submit;\r\n3).Appear Error : This Approval\
    \ Request Requires The Next Approver To Be Determined By The Discount Approver\
    \ Field. This Value Is Empty. Please Contact Your Administrator For More Information;\r\
    \nFor Detail Please see ServiceNow Attachment and My E-mail Too Action Plan taken\
    \ by AMS: We have updated the discount approver name to the WO-PGI-WO-2501006104,"
  sentences:
  - 'INC0043354 Dear Team


    Error occurred while processing the EDI transaction


    Interface       internal-stocks

    Subsidiary      PAU

    API Name        pana-sdesk-ext-eapi

    Flow Direction  outbound

    Source System   PAU

    End System      ZOHO

    File Name       No File Name

    Storage Path    No Attachment

    Error Source    pana-sdesk-ext-eapi

    Transaction ID  f489f3a0-aaec-11ef-8b0c-02b4130d4440

    Error Summary   500 SOURCE_RESPONSE_SEND

    Error Details   SOURCE_RESPONSE_SEND: ''/Zoho/Live/Outbound/status/remote/SPR_Status_20241121090803362047.csv''
    cannot be renamed because ''/Zoho/Live/Outbound/status/done/SPR_Status_20241121090803362047.csv''
    already exists

    Comments


    Note: This is an automated mail, please do not reply.


    Thanks and Regards,


    APAC Support, Panasonic The setup for the file poller in outbound servicedesk
    is it does not do overwrite, thus, the issue encountered. As a workaround, the
    previous file has been deleted. As for the resolution, overwrite is now enabled.'
  - "INC0059006 User Check Out could not be processed. Detail: There was a problem\
    \ connecting to the SAP System.  please check on this issue urgently as we have\
    \ some urgent issue need to be settled ASAP. Issue: User unable to release Power\
    \ ID from soterian\r\nCategory: Issue\r\nFailure Point: NA\r\nRoot Cause: NA\r\
    \nFix: This was a temporary issue and it got resolved after few mins. The users\
    \ were able to release and approve the Power IDs.\r\nPreventive Action: Manually\
    \ power IDs can be released if there is issue in future.\r\nOutage: NA\r\nCR/Story:\
    \ NA\r\nKnowledge Object: NA\r\nProblem Ticket: NA\r\nClosure Evidence Attached\
    \ (Y/N): Yes"
  - 'INC0055627 Dear Team


    Error occurred while processing the EDI transaction


    Interface

    Subsidiary      No Subsidiary

    API Name        pana-pagitp-mgmt-eapi

    Flow Direction  Inbound

    Source System   PAGITP

    End System      SAP S4Hana

    File Name       No File Name

    Storage Path    No Attachment

    Error Source    Mulesoft

    Transaction ID  cc831d9d-3e68-4a14-8e9d-eeed75ac682c

    Error Summary   500 SOURCE_RESPONSE_SEND

    Error Details   Client connection was closed

    Comments


    Note: This is an automated mail, please do not reply.


    Thanks and Regards,


    APAC Support, Panasonic No action needed. Data successfully sent to SAP.'
- source_sentence: "INC0038871 SIP prices reflect differently on both admin & creator's\
    \ account. Unable to process PO, and unable to find the reason. Issue: SIP not\
    \ updated in Sales orders\r\nCause: User uploaded new price hence different price\
    \ picked up\r\nSolution: user upload new correct SIP"
  sentences:
  - "INC0035358 Hi Ganesh & team,\r\n\r\nWe currently have many cases from March/2024\
    \ where we see Salesforce transactions created by SOA with the type 'Adjusted'\
    \ related to stock (non-refrigerant related) but can't find the related record\
    \ on the same date in SAP to know where it's sourcing from.\r\n\r\nWe have an\
    \ inquiry on how much effort would consist to map from SAP to SOA the Material\
    \ Document number from documents created in SAP for records such as the example\
    \ above.\r\n\r\nCurrently, it's been hard to track how some of these records have\
    \ been created and from what source in SAP, as the only thing we see is the related\
    \ item and Transaction Type as Adjusted, but no clue what document is pushing\
    \ this change from SAP to SOA.\r\n\r\nWe can't know if it was due to a transfer,\
    \ a goods issue, etc.\r\n\r\nWe don't need to know the type of transaction. \r\
    \nAs long as we have the document number, we'll be okay and won't get into the\
    \ situation where we need to track flat files for transactions that happened at\
    \ a time when we no longer have backups\r\n\r\nAttached is an example of a document\
    \ we could find the source in SAP\r\n\r\nThank you For effort estimation, requested\
    \ Biz to raise RFP. However, understand Biz wanted to drop this request and close.\r\
    \nRegards to TR movement, raised Change request CHG0031108 accordingly to fix\
    \ the Job."
  - "INC0062628 When customer reset their password for Auth0 Account, the below message\
    \ appear which is Malaysia Customer Portal \r\n\r\n\"Your password has been reset\
    \ for Malaysia Customer Portal Your password has been reset for Malaysia Customer\
    \ Portal - CS. Go to:\r\n\r\n            https://p-cube.panasonic.com/malaysia/login?c=0nhw6SsLniQC8EWS_YU8VvVxuIes0Xg.rhsMdjKOlTTzVJ4RqQvsEidQN0EAxBijf5flp6N2dNzvMuhI_xzntfU.oEVAqOeU3oMQwbdhL59ebUodD3XJeqgSKyHYPqgXwZGK9RsjZBBGmLjixUgd.3Dsdpw0L79LEuHXnbqOH7krXFOhbogRL9gAXawExv5a9X1oMmYKjjISiGJzPNHV3gD_I9kkaA%3D%3D\
    \ Resolution Status: Closed (Not an Incident)\r\n\r\nAdditional Update: Account\
    \ updated under RITM0066803."
  - "INC0020744 Item Part F0395CS80HP, F4440BY00XP,  F603LCS80TT,  F605QCS80HP,  F6141CS00XP,\
    \ F6145-11A0, F6145CF00XP, F6437-1S30 & F8337CS80HP   is cannot be uploaded to\
    \ the DMR, When we try to Upload DMR Template is appear notification \"Enter a\
    \ valid material\" Message No. /IRM/EPG107,\r\nFor detail Please see my email\
    \ and ITMAAS (ServiceNow) Attachment Closed by Caller"
pipeline_tag: sentence-similarity
library_name: sentence-transformers
metrics:
- pearson_cosine
- spearman_cosine
model-index:
- name: SentenceTransformer based on sentence-transformers/all-mpnet-base-v2
  results:
  - task:
      type: semantic-similarity
      name: Semantic Similarity
    dataset:
      name: eval subset
      type: eval_subset
    metrics:
    - type: pearson_cosine
      value: .nan
      name: Pearson Cosine
    - type: spearman_cosine
      value: .nan
      name: Spearman Cosine
---

# SentenceTransformer based on sentence-transformers/all-mpnet-base-v2

This is a [sentence-transformers](https://www.SBERT.net) model finetuned from [sentence-transformers/all-mpnet-base-v2](https://huggingface.co/sentence-transformers/all-mpnet-base-v2). It maps sentences & paragraphs to a 768-dimensional dense vector space and can be used for semantic textual similarity, semantic search, paraphrase mining, text classification, clustering, and more.

## Model Details

### Model Description
- **Model Type:** Sentence Transformer
- **Base model:** [sentence-transformers/all-mpnet-base-v2](https://huggingface.co/sentence-transformers/all-mpnet-base-v2) <!-- at revision e8c3b32edf5434bc2275fc9bab85f82640a19130 -->
- **Maximum Sequence Length:** 256 tokens
- **Output Dimensionality:** 768 dimensions
- **Similarity Function:** Cosine Similarity
<!-- - **Training Dataset:** Unknown -->
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Documentation:** [Sentence Transformers Documentation](https://sbert.net)
- **Repository:** [Sentence Transformers on GitHub](https://github.com/huggingface/sentence-transformers)
- **Hugging Face:** [Sentence Transformers on Hugging Face](https://huggingface.co/models?library=sentence-transformers)

### Full Model Architecture

```
SentenceTransformer(
  (0): Transformer({'max_seq_length': 256, 'do_lower_case': False, 'architecture': 'PeftModelForFeatureExtraction'})
  (1): Pooling({'word_embedding_dimension': 768, 'pooling_mode_cls_token': False, 'pooling_mode_mean_tokens': True, 'pooling_mode_max_tokens': False, 'pooling_mode_mean_sqrt_len_tokens': False, 'pooling_mode_weightedmean_tokens': False, 'pooling_mode_lasttoken': False, 'include_prompt': True})
  (2): Normalize()
)
```

## Usage

### Direct Usage (Sentence Transformers)

First install the Sentence Transformers library:

```bash
pip install -U sentence-transformers
```

Then you can load this model and run inference.
```python
from sentence_transformers import SentenceTransformer

# Download from the 🤗 Hub
model = SentenceTransformer("sentence_transformers_model_id")
# Run inference
sentences = [
    "INC0038871 SIP prices reflect differently on both admin & creator's account. Unable to process PO, and unable to find the reason. Issue: SIP not updated in Sales orders\r\nCause: User uploaded new price hence different price picked up\r\nSolution: user upload new correct SIP",
    'INC0020744 Item Part F0395CS80HP, F4440BY00XP,  F603LCS80TT,  F605QCS80HP,  F6141CS00XP, F6145-11A0, F6145CF00XP, F6437-1S30 & F8337CS80HP   is cannot be uploaded to the DMR, When we try to Upload DMR Template is appear notification "Enter a valid material" Message No. /IRM/EPG107,\r\nFor detail Please see my email and ITMAAS (ServiceNow) Attachment Closed by Caller',
    'INC0062628 When customer reset their password for Auth0 Account, the below message appear which is Malaysia Customer Portal \r\n\r\n"Your password has been reset for Malaysia Customer Portal Your password has been reset for Malaysia Customer Portal - CS. Go to:\r\n\r\n            https://p-cube.panasonic.com/malaysia/login?c=0nhw6SsLniQC8EWS_YU8VvVxuIes0Xg.rhsMdjKOlTTzVJ4RqQvsEidQN0EAxBijf5flp6N2dNzvMuhI_xzntfU.oEVAqOeU3oMQwbdhL59ebUodD3XJeqgSKyHYPqgXwZGK9RsjZBBGmLjixUgd.3Dsdpw0L79LEuHXnbqOH7krXFOhbogRL9gAXawExv5a9X1oMmYKjjISiGJzPNHV3gD_I9kkaA%3D%3D Resolution Status: Closed (Not an Incident)\r\n\r\nAdditional Update: Account updated under RITM0066803.',
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 768]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities)
# tensor([[1.0000, 0.2263, 0.2310],
#         [0.2263, 1.0000, 0.3240],
#         [0.2310, 0.3240, 1.0000]])
```

<!--
### Direct Usage (Transformers)

<details><summary>Click to see the direct usage in Transformers</summary>

</details>
-->

<!--
### Downstream Usage (Sentence Transformers)

You can finetune this model on your own dataset.

<details><summary>Click to expand</summary>

</details>
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

## Evaluation

### Metrics

#### Semantic Similarity

* Dataset: `eval_subset`
* Evaluated with [<code>EmbeddingSimilarityEvaluator</code>](https://sbert.net/docs/package_reference/sentence_transformer/evaluation.html#sentence_transformers.evaluation.EmbeddingSimilarityEvaluator)

| Metric              | Value   |
|:--------------------|:--------|
| pearson_cosine      | nan     |
| **spearman_cosine** | **nan** |

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Dataset

#### Unnamed Dataset

* Size: 5,000 training samples
* Columns: <code>sentence_0</code>, <code>sentence_1</code>, and <code>label</code>
* Approximate statistics based on the first 1000 samples:
  |         | sentence_0                                                                           | sentence_1                                                                           | label                                                          |
  |:--------|:-------------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------|:---------------------------------------------------------------|
  | type    | string                                                                               | string                                                                               | float                                                          |
  | details | <ul><li>min: 13 tokens</li><li>mean: 136.79 tokens</li><li>max: 256 tokens</li></ul> | <ul><li>min: 12 tokens</li><li>mean: 135.35 tokens</li><li>max: 256 tokens</li></ul> | <ul><li>min: 0.0</li><li>mean: 0.54</li><li>max: 1.0</li></ul> |
* Samples:
  | sentence_0                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  | sentence_1                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 | label            |
  |:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-----------------|
  | <code>INC0021293 DMR post fail Resolved by SR REQ0019748</code>                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             | <code>INC0054045 Dear Team<br><br>Error occurred while processing the EDI transaction<br><br>Interface       attendance-swipes<br>Subsidiary      PIDMY<br>API Name        pana-hriq-int-eapi<br>Flow Direction  inbound<br>Source System   PIDMY<br>End System      ZingHR<br>File Name       NA<br>Storage Path    No Attachment<br>Error Source    ZingHR<br>Transaction ID  922a893a-4ff9-4ad3-ac7c-2f78923bd46c<br>Error Summary   503 SERVICE_UNAVAILABLE<br>Error Details   ***********443/api/v1/ZingHR/employees/attendance-swipe' failed: Timeout exceeded."<br>Comments<br><br>Note: This is an automated mail, please do not reply.<br><br>Thanks and Regards,<br><br>APAC Support, Panasonic No actions needed</code>                                                                                                                         | <code>1.0</code> |
  | <code>INC0051745 User: Irene  <br>Pls kindly help to check why she is unable to submit the below eClaim. Pls check urgently Issue occurred because section fund balance is not correct.</code>                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               | <code>INC0056529 Please change the UOM for part # ACXE22K06981 in PIR from SET to PC so that we can please PO's for this part. IT incident ticket requested by Pandurang Mane Issue: PNP 900 system unable to place PO with PC UoM.  <br>Category:  SAP MM  <br>Failure Point:  PO creation  <br>Root Cause:  The UoM in PO has picked from PIR and due to the PC UoM has not maintained in PIR hence system was not allowed to create PO with PC UoM.  <br>Fix: Sanity check from business to create PO in PNP 900 system.  <br>Preventive Action: Sanity check performed by business.  <br>Outage: NA  <br>CR/Story: NA  <br>Knowledge Object: NA  <br>Problem Ticket: NA  <br>Closure Evidence Attached (Y/N): NA</code>                                                                                                                                          | <code>1.0</code> |
  | <code>INC0020015 when many WOs are created in the same time by Call Agents, system  frequently shows  "duplicate value found". It's takes long time to wait and cant save. Its affects seriously to Call center's work performance. Kindly help to fix in urgent From: Le Thi Thu Huyen <thuhuyen.le@vn.panasonic.com>   <br>Sent: Thursday, June 6, 2024 2:32 PM  <br>To: Nguyen Trung Thuc <trungthuc.nguyen@vn.panasonic.com>; Mohammad Arif <mohammad.arif@sg.panasonic.com>  <br>Cc: Dinesh Gatla <dinesh.gatla@sg.panasonic.com>  <br>Subject: RE: Incident INC0020015 has been assigned to group CAPG L2 CRM  <br>  <br>Hi Team  <br>                Thank you for quick support. This below issue has been resolved completely. Pls close the ticket.  <br>  <br>Regards,  <br>Thu Huyen  <br>  <br>  <br>From: Nguyen Trung Thuc <trungthuc.nguyen@vn.panasonic.com>   <br>Sent: Thursday, June 6, 2024 3:20 PM  <br>To: Mohammad Arif <mohammad.arif@sg.panasonic.com>; Le Thi Thu Huyen <thuhuyen.le@vn.panasonic.com>  <br>Cc: Dinesh Gatla <dinesh.gatla@sg.panasonic.com>  <br>Subject: RE: Incident...</code> | <code>INC0040454 Dear Team<br><br>Error occurred while processing the EDI transaction<br><br>Interface       raw-doucments<br>Subsidiary      PAVCKM<br>API Name        pana-my-einvoice-mgmt-papi<br>Flow Direction  outbound<br>Source System   PAVCKM<br>End System      IRBM<br>File Name       No File Name<br>Storage Path    No Attachment<br>Error Source    pana-my-einvoice-mgmt-papi<br>Transaction ID  c0ebd281-9303-11ef-a3b9-aa80b05d85a3<br>Error Summary   403 RETRY_EXHAUSTED<br>Error Details   ***********443/api/v1/PAVCKM/eInvoice/documents/MTNSJ28FSN8DNWG2PHH261BJ10/raw' failed: forbidden (403).<br>Comments<br><br>Note: This is an automated mail, please do not reply.<br><br>Thanks and Regards,<br><br>APAC Support, Panasonic this is due to forbidden error on the IRB server side, user is alr informed via email</code> | <code>1.0</code> |
* Loss: [<code>CosineSimilarityLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#cosinesimilarityloss) with these parameters:
  ```json
  {
      "loss_fct": "torch.nn.modules.loss.MSELoss"
  }
  ```

### Training Hyperparameters
#### Non-Default Hyperparameters

- `eval_strategy`: steps
- `per_device_train_batch_size`: 32
- `per_device_eval_batch_size`: 32
- `num_train_epochs`: 10
- `multi_dataset_batch_sampler`: round_robin

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `overwrite_output_dir`: False
- `do_predict`: False
- `eval_strategy`: steps
- `prediction_loss_only`: True
- `per_device_train_batch_size`: 32
- `per_device_eval_batch_size`: 32
- `per_gpu_train_batch_size`: None
- `per_gpu_eval_batch_size`: None
- `gradient_accumulation_steps`: 1
- `eval_accumulation_steps`: None
- `torch_empty_cache_steps`: None
- `learning_rate`: 5e-05
- `weight_decay`: 0.0
- `adam_beta1`: 0.9
- `adam_beta2`: 0.999
- `adam_epsilon`: 1e-08
- `max_grad_norm`: 1
- `num_train_epochs`: 10
- `max_steps`: -1
- `lr_scheduler_type`: linear
- `lr_scheduler_kwargs`: {}
- `warmup_ratio`: 0.0
- `warmup_steps`: 0
- `log_level`: passive
- `log_level_replica`: warning
- `log_on_each_node`: True
- `logging_nan_inf_filter`: True
- `save_safetensors`: True
- `save_on_each_node`: False
- `save_only_model`: False
- `restore_callback_states_from_checkpoint`: False
- `no_cuda`: False
- `use_cpu`: False
- `use_mps_device`: False
- `seed`: 42
- `data_seed`: None
- `jit_mode_eval`: False
- `bf16`: False
- `fp16`: False
- `fp16_opt_level`: O1
- `half_precision_backend`: auto
- `bf16_full_eval`: False
- `fp16_full_eval`: False
- `tf32`: None
- `local_rank`: 0
- `ddp_backend`: None
- `tpu_num_cores`: None
- `tpu_metrics_debug`: False
- `debug`: []
- `dataloader_drop_last`: False
- `dataloader_num_workers`: 0
- `dataloader_prefetch_factor`: None
- `past_index`: -1
- `disable_tqdm`: False
- `remove_unused_columns`: True
- `label_names`: None
- `load_best_model_at_end`: False
- `ignore_data_skip`: False
- `fsdp`: []
- `fsdp_min_num_params`: 0
- `fsdp_config`: {'min_num_params': 0, 'xla': False, 'xla_fsdp_v2': False, 'xla_fsdp_grad_ckpt': False}
- `fsdp_transformer_layer_cls_to_wrap`: None
- `accelerator_config`: {'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None}
- `parallelism_config`: None
- `deepspeed`: None
- `label_smoothing_factor`: 0.0
- `optim`: adamw_torch_fused
- `optim_args`: None
- `adafactor`: False
- `group_by_length`: False
- `length_column_name`: length
- `project`: huggingface
- `trackio_space_id`: trackio
- `ddp_find_unused_parameters`: None
- `ddp_bucket_cap_mb`: None
- `ddp_broadcast_buffers`: False
- `dataloader_pin_memory`: True
- `dataloader_persistent_workers`: False
- `skip_memory_metrics`: True
- `use_legacy_prediction_loop`: False
- `push_to_hub`: False
- `resume_from_checkpoint`: None
- `hub_model_id`: None
- `hub_strategy`: every_save
- `hub_private_repo`: None
- `hub_always_push`: False
- `hub_revision`: None
- `gradient_checkpointing`: False
- `gradient_checkpointing_kwargs`: None
- `include_inputs_for_metrics`: False
- `include_for_metrics`: []
- `eval_do_concat_batches`: True
- `fp16_backend`: auto
- `push_to_hub_model_id`: None
- `push_to_hub_organization`: None
- `mp_parameters`: 
- `auto_find_batch_size`: False
- `full_determinism`: False
- `torchdynamo`: None
- `ray_scope`: last
- `ddp_timeout`: 1800
- `torch_compile`: False
- `torch_compile_backend`: None
- `torch_compile_mode`: None
- `include_tokens_per_second`: False
- `include_num_input_tokens_seen`: no
- `neftune_noise_alpha`: None
- `optim_target_modules`: None
- `batch_eval_metrics`: False
- `eval_on_start`: False
- `use_liger_kernel`: False
- `liger_kernel_config`: None
- `eval_use_gather_object`: False
- `average_tokens_across_devices`: True
- `prompts`: None
- `batch_sampler`: batch_sampler
- `multi_dataset_batch_sampler`: round_robin
- `router_mapping`: {}
- `learning_rate_mapping`: {}

</details>

### Training Logs
| Epoch | Step | eval_subset_spearman_cosine |
|:-----:|:----:|:---------------------------:|
| 1.0   | 157  | nan                         |


### Framework Versions
- Python: 3.11.14
- Sentence Transformers: 5.2.0
- Transformers: 4.57.3
- PyTorch: 2.11.0.dev20251223+cu128
- Accelerate: 1.12.0
- Datasets: 4.4.1
- Tokenizers: 0.22.1

## Citation

### BibTeX

#### Sentence Transformers
```bibtex
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084",
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->