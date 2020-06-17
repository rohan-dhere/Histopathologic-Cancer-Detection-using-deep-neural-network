<?php
include "../php/config.php";
session_start();
$user= $_SESSION["loggedin_user"];

$sql="SELECT * from patients WHERE email_id='".$_SESSION["loggedin_user"]."'";
$result=$conn->query($sql);
while ($row=mysqli_fetch_assoc($result))
{
    $_SESSION["id"]=$row['id'];
    $_SESSION["name"] = $row['name'];
    $_SESSION["age"] = $row['age'];
    $_SESSION["dob"] = $row['date of birth'];
    $_SESSION["sex"] = $row['sex'];
    $_SESSION["blood_group"] = $row['blood_group'];
    $_SESSION["contact"] = $row['contact'];
    $_SESSION["address"] = $row['address'];
    $_SESSION["medical_history"] = $row['medical_history'];
    $_SESSION["email"] = $row['email_id'];
    
}



$sql2="SELECT * from report WHERE email='".$_SESSION["loggedin_user"]."'";
$result2=$conn->query($sql2);

while ($row=mysqli_fetch_assoc($result2))
{
    
    $_SESSION["path"] = $row['path'];
    $_SESSION["date"] = $row['date'];
    $_SESSION["time"] = $row['time'];
    $_SESSION["result"] = $row['result'];
    
}



$conn->close();




?>









<!DOCTYPE html>
<html>

<head>

    <meta charset="utf-8">
    <title>Report</title>
    <link rel="stylesheet" href="../css/report.css" media="all" />



</head>

<body class="body1">
     
        <header class="header1">
            <div class="clearfix">
                <div id="logo">
                    <img src="../images/144315.svg">
                </div>
                <h1>Medical Report</h1>
                <div id="company" class="clearfix">
                    <div>APSIT HEALTH CENTER</div>
                    <div>Ghodbunder Rd, opp. Hypercity Mall,<br /> Kasarvadavali, Thane West, Thane,</div>
                    <div>Maharashtra 400615</div>
                    <div>+91-123456789</div>
                    <div><a href="mailto:company@example.com">apsitmlproject@gmail.com</a></div>
                </div>
                <div id="project">
                    <div><span>Patitent Name </span>   <?php echo $_SESSION["name"] ; ?> </div>
                    <div><span>Address <Address</span> <?php echo $_SESSION["address"] ; ?> </div>
                    <div><span>DATE Of Birth </span>   <?php echo $_SESSION["dob"] ; ?></div>
                    
                    <div><span>Report Date </span>     <?php echo  $_SESSION["date"];?></div>
                    
                    <div><span>Report Time </span>     <?php echo  $_SESSION["time"];?></div>
                </div>
            </div>
    </header>
    <hr>
    <h3>Histopathologic Cancer Report For Patitent : <?php echo $_SESSION["id"] ; ?></h3>
    <div> Clinical Diagnosis : <?php echo  $_SESSION["result"]; ?> </div>

    <div> Date of Diagnosis : <?php echo  $_SESSION["date"];?> </div>

    <div><h4> Gross Description: <h4> </div>
    <div><p>The specimen is received in two parts. They are labeled #1, "biopsy bladder tumor", and #2, "scalene node, left". 
        Part #1 consists of multiple fragments of gray-brown tissue which appear slightly hemorrhagic. 
        They are submitted in their entirety for processing. Part #2 consists of multiple fragments of fatty yellow tissue 
        which range in size from 0.2 to 1.0 cm in diameter. 
        They are submitted in their entirety for processing.</p></div>
    
        
    <div><h4>Microscopic:  <h4> </div>
    <div> <p>Section of bladder contains areas of transitional cell carcinoma. 
    No area of invasion can be identified. A marked acute and chronic inflammatory 
    reaction with eosinophils is noted together with some necrosis. Sections are examined at six levels. 
    Section of lymph node contains normal node with reactive germinal centers.All sections taken radially 
    from the superficial center of the resection site fail to include tumor, indicating the tumor to have 
    originated deep within the breast parenchyma. Similarly, there is no malignancy in the nipple region, 
    or in the lactiferous sinuses.Sections of deep surgical margin demonstrate diffuse tumor infiltration
    of deep fatty tissues, however, there is no invasion of muscle. Total size of primary tumor is estimated 
    to be 4 cm in greatest dimension.</p> </div>

    <div>
    </div>


    
    <div class="footer">
    
    Report was created on a computer and is valid without the signature and seal.
    <br>
        <button onclick="printinvoice()">Print Report</button>
    <a href="../upload_img.php">Home</a>
    </div>
    <script>
            function printinvoice()
            {
                window.print();
            }

    </script>
    
    </div>






</body>

</html>





























